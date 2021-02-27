## Vizi-AI Kit

*Connected accordingly*

Initially I couldn't discover the device I thought a driver is required or so, but none, I had to just restart the ADLINK Edge Profiler Platform on PC. 
Recollect, the SD card also runs the profiler platform.

Then I found the device needs updating / its platform

*Vizi-AI Kit*

- Node-RED was amended (the code)

- Then the model-confidence regex value was changed to eliminate the log error

- Licensing was found tobe a requirment for openvino and frame-streamer

- there is a SOAP server running:

<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/" xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:cms="http://127.0.0.1/cms.wsdl">
<SOAP-ENV:Body>
<SOAP-ENV:Fault>
<faultcode>SOAP-ENV:Client</faultcode>
<faultstring>HTTP GET method not implemented</faultstring>
</SOAP-ENV:Fault>
</SOAP-ENV:Body>
</SOAP-ENV:Envelope>

*Size Limitation*

- 32GB is flowing out due to docker size

- There is a vizi-ai-starter-kit installation profile from ADLINK, this is causing several outages in size limitations.


### Some output images of inference

![./inference_outputs/img1.png](./inference_outputs/img1.png)

![./inference_outputs/img2.png](./inference_outputs/img2.png)

![./inference_outputs/img3.png](./inference_outputs/img3.png)

![./inference_outputs/img4.png](./inference_outputs/img4.png)
