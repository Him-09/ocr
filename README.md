# Bolt Hole Identifier Model

A deep learning model for recognizing numbers in a bolt hole images, with special handling for cases where no number is present.

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

## Monitoring and Maintenance

1. View container logs:
```bash
docker logs <container_id>
```

2. Restart container:
```bash
docker restart <container_id>
```

3. Update application:
```bash
git pull
docker build -t license-plate-api .
docker stop <container_id>
docker rm <container_id>
docker run -d -p 8000:8000 license-plate-api
```

## Security Considerations

1. Keep system and dependencies updated
2. Use HTTPS for all API calls
3. Implement rate limiting
4. Regular security audits
5. Monitor logs for suspicious activity

## Troubleshooting

1. Check container status:
```bash
docker ps
```

2. View container logs:
```bash
docker logs <container_id>
```

3. Check Nginx logs:
```bash
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log
```

4. Verify SSL certificate:
```bash
certbot certificates
```
