# License Plate Number Recognition API

This is a FastAPI-based service for recognizing numbers in license plate images. The service uses a deep learning model trained on license plate images to identify numbers and detect when no number is present.

## API Endpoints

- `GET /`: Root endpoint with API information
- `POST /predict`: Endpoint for predicting numbers in license plate images
- `GET /health`: Health check endpoint

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t license-plate-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 license-plate-api
```

## DigitalOcean Deployment

1. Create a new Droplet on DigitalOcean:
   - Choose Ubuntu 20.04 LTS
   - Select a plan with at least 2GB RAM
   - Choose a datacenter region
   - Add your SSH key

2. SSH into your Droplet:
```bash
ssh root@your_droplet_ip
```

3. Install Docker:
```bash
# Update system
apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Add your user to docker group
usermod -aG docker $USER
```

4. Clone the repository:
```bash
git clone your_repository_url
cd your_repository
```

5. Build and run the Docker container:
```bash
docker build -t license-plate-api .
docker run -d -p 8000:8000 license-plate-api
```

6. Set up Nginx as reverse proxy (optional but recommended):
```bash
# Install Nginx
apt-get install nginx

# Create Nginx configuration
cat > /etc/nginx/sites-available/license-plate-api << 'EOL'
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
EOL

# Enable the site
ln -s /etc/nginx/sites-available/license-plate-api /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default

# Test and restart Nginx
nginx -t
systemctl restart nginx
```

7. Set up SSL with Certbot (optional but recommended):
```bash
# Install Certbot
apt-get install certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d your_domain.com
```

## API Usage

### Predict Endpoint

Send a POST request to `/predict` with an image file:

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://your_domain.com/predict
```

Response format:
```json
{
    "predicted_number": "42",
    "confidence": 0.98,
    "raw_probabilities": [...]
}
```

### Health Check

Check the API health:

```bash
curl http://your_domain.com/health
```

Response format:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
}
```

## Monitoring and Maintenance

1. Check container logs:
```bash
docker logs license-plate-api
```

2. Restart the container:
```bash
docker restart license-plate-api
```

3. Update the application:
```bash
git pull
docker build -t license-plate-api .
docker stop license-plate-api
docker run -d -p 8000:8000 license-plate-api
```

## Security Considerations

1. Set up a firewall:
```bash
ufw allow ssh
ufw allow http
ufw allow https
ufw enable
```

2. Use environment variables for sensitive data
3. Regularly update the system and dependencies
4. Monitor logs for suspicious activity

## Troubleshooting

1. Check container status:
```bash
docker ps
```

2. View container logs:
```bash
docker logs license-plate-api
```

3. Check Nginx logs:
```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

4. Restart services:
```bash
docker restart license-plate-api
systemctl restart nginx
``` 