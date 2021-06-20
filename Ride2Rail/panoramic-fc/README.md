# Panoramic Feature Collector
In order to run this Feature Collector, you need to store the data into the [cache][routerank-extractor]

## Note
In order to run this Feature Collector, you need to be connected to Eurecat VPN. The data from the different countries
has been mapped and stored in Docker containers from [OpenStreetMap][osm] in Eurecat servers. 

## Usage

### Local development (debug on)

```bash
$ python3 panoramic.py
 * Serving Flask app "panoramic" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 262-589-651
```

### Local development with Docker

```bash
docker run \
  --rm \
  -it \
  --name panoramic \
  -p 5000:5000 \
  --link cache:cache \
  -e FLASK_ENV='development' \
  -v "$PWD":/code \
    r2r/panoramic-fc:latest
```

```bash
$ FLASK_APP='panoramic.py' flask run)
 * Serving Flask app "panoramic.py"
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

## Example Request

```bash
$ curl --header 'Content-Type: application/json' \
       --request POST  \
       --data '{"request_id": "123x" }' \
         http://localhost:5000/compute
{"request_id": "123x"}%
```
[routerank-extractor]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/Ride2Rail/routerank-extractor
[osm]: https://www.openstreetmap.org