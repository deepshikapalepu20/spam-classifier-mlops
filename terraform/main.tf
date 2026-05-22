terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "default"
}

# Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# Default VPC
data "aws_vpc" "default" {
  default = true
}

# Security Group
resource "aws_security_group" "spam_classifier_sg" {

  name        = "spam-classifier-sg"
  description = "Allow SSH and Flask App"
  vpc_id      = data.aws_vpc.default.id

  # SSH

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Flask Application

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Optional Prometheus

  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Optional Grafana

  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {

    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]

  }

  tags = {
    Name = "SpamClassifier-SG"
  }

}

# EC2 Deployment Server

resource "aws_instance" "deployment_server" {

  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type

  vpc_security_group_ids = [
    aws_security_group.spam_classifier_sg.id
  ]

  user_data = <<-EOF
#!/bin/bash

apt-get update -y

# Install Docker

apt-get install -y docker.io

systemctl start docker
systemctl enable docker

usermod -aG docker ubuntu

# Install Docker Compose

curl -L \
"https://github.com/docker/compose/releases/download/v2.24.6/docker-compose-linux-x86_64" \
-o /usr/local/bin/docker-compose

chmod +x /usr/local/bin/docker-compose

# Pull your Docker image

docker pull YOUR_DOCKERHUB_USERNAME/spam-classifier:latest

# Run container

docker run -d \
-p 5000:5000 \
--name spam-classifier \
YOUR_DOCKERHUB_USERNAME/spam-classifier:latest

EOF

  tags = {

    Name = "SpamClassifier-Deployment-Server"
    Project = "SpamClassifier"

  }

}