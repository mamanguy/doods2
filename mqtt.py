import uvicorn
import json
import base64
import logging
import threading
import time
import cv2
import paho.mqtt.client as mqtt
import numpy as np
import os
import traceback
from streamer import Streamer
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class fshandler(FileSystemEventHandler):
    def __init__(self, doods, detect_request, mqtt, logger=None):
        super().__init__()
        self.doods = doods
        self.detect_request = detect_request
        self.mqtt = mqtt
        self.logger = logger or logging.root


    def on_created(self, event):
        if event.is_directory == True:
            return
        try:
            self.detect_request.filename = os.path.basename(event.src_path)
            with open(event.src_path, "rb") as f:
                self.detect_request.data = base64.b64encode(f.read()).decode("utf-8") 
            detect_response = self.doods.detect(self.detect_request)
            if detect_response.error:
                print ("Error")
            else:
                self.mqtt.publish_response(self.detect_request, detect_response)
        except Exception as e:
            self.logger.error(e)
            traceback.print_exc()
            pass

class MQTT():
    def __init__(self, config, doods, metrics_server_config=None):
        self.config = config
        self.doods = doods
        self.metrics_server_config = metrics_server_config
        self.mqtt_client = mqtt.Client()
        self.app = FastAPI()
        # Borrow the uvicorn logger because it's pretty.
        self.logger = logging.getLogger("doods.mqtt")

        @self.app.get("/annotations")
        async def annotations(image_request: str, response: Response):
            image = None
            with open (f"/annotations/{image_request}", 'rb') as f:
                image = f.read()
            return Response(content=image, media_type="image/jpeg")

    def publish_response(self, mqtt_detect_request, detect_response):
        # If separate_detections, iterate over each detection and process it separately
        if mqtt_detect_request.separate_detections:

            #If we're going to be cropping, do this processing only once (rather than for each detection)
            if mqtt_detect_request.image and mqtt_detect_request.crop:
                detect_image_bytes = np.frombuffer(detect_response.image, dtype=np.uint8)
                detect_image = cv2.imdecode(detect_image_bytes, cv2.IMREAD_COLOR)
                di_height, di_width = detect_image.shape[:2]


            for detection in detect_response.detections:
                # If an image was requested
                if mqtt_detect_request.image:
                    # Crop image to detection box if requested
                    if mqtt_detect_request.crop:
                        cropped_image = detect_image[
                        int(detection.top*di_height):int(detection.bottom*di_height), 
                        int(detection.left*di_width):int(detection.right*di_width)]
                        mqtt_image = cv2.imencode(mqtt_detect_request.image, cropped_image)[1].tostring()
                    else:
                        mqtt_image = detect_response.image


                    # For binary images, publish the image to its own topic
                    if mqtt_detect_request.binary_images:
                        self.mqtt_client.publish(
                            f"doods/image/{mqtt_detect_request.id}{'' if detection.region_id is None else '/'+detection.region_id}/{detection.label or 'object'}", 
                            payload=mqtt_image, qos=0, retain=False)
                    # Otherwise add base64-encoded image to the detection
                    else:
                        detection.image = base64.b64encode(mqtt_image).decode('utf-8')

                self.mqtt_client.publish(
                    f"doods/detect/{mqtt_detect_request.id}{'' if detection.region_id is None else '/'+detection.region_id}/{detection.label or 'object'}", 
                    payload=json.dumps(detection.asdict(include_none=False)), qos=0, retain=False)

        # Otherwise, publish the collected detections together
        else:
            # If an image was requested
            if mqtt_detect_request.image:
                if mqtt_detect_request.save_local:
                    with open (f"/annotations/{mqtt_detect_request.filename}", 'wb') as f:
                        f.write(detect_response.image)
                        detect_response.image = ''
                else:
                    # If binary_images, move the image from the response and publish it to a separate topic
                    if mqtt_detect_request.binary_images:
                        mqtt_image = detect_response.image
                        detect_response.image = None
                        self.mqtt_client.publish(
                            f"doods/image/{mqtt_detect_request.id}", 
                            payload=detect_response.image, qos=0, retain=False)
                    # Otherwise, inlcude the base64-encoded image in the response
                    else:
                        detect_response.image = base64.b64encode(detect_response.image).decode('utf-8')
            
            self.mqtt_client.publish(
                    f"doods/detect/{mqtt_detect_request.id}", 
                    payload=json.dumps(detect_response.asdict(include_none=False)), qos=0, retain=False)
        

    def stream(self, mqtt_detect_request: str = '{}'):
        streamer = None
        try:
            # Run the stream detector and return the results.
            streamer = Streamer(self.doods).start_stream(mqtt_detect_request)
            for detect_response in streamer:
                self.publish_response(mqtt_detect_request, detect_response)
                    

        except Exception as e:
            self.logger.info(e)
            try:
                if streamer:
                    streamer.send(True)  # Stop the streamer
            except StopIteration:
                pass

    def directory_listener(self, mqtt_detect_request: str = '{}'):
        loop = True
        while (loop):
            try:
                event_handler = fshandler(self.doods, mqtt_detect_request, self) #LoggingEventHandler()
                observer = Observer()
                observer.schedule(event_handler, mqtt_detect_request.watching_directory, recursive=True)
                observer.start()
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    observer.stop()
                    loop = False
                observer.join()
            except Exception as e:
                self.logger.error(e)
                traceback.print_exc()
                pass
        
    def metrics_server(self, config):
        self.instrumentator = Instrumentator(
            excluded_handlers=["/metrics"],
        )
        self.instrumentator.instrument(self.app).expose(self.app)
        uvicorn.run(self.app, host=config.host, port=config.port, log_config=None)

    def run(self):
        if (self.config.broker.user):
            self.mqtt_client.username_pw_set(self.config.broker.user, self.config.broker.password)
        self.mqtt_client.connect(self.config.broker.host, self.config.broker.port, 60)

        for request in self.config.requests:
            if request.watching_directory == "":
                self.logger.info("reading stream")
                threading.Thread(target=self.stream, args=(request,)).start()
            else:    
                self.logger.info("watching directory")
                threading.Thread(target=self.directory_listener, args=(request,)).start()

        if self.config.metrics:
            self.logger.info("starting metrics server")
            self.metrics_server(self.metrics_server_config)


