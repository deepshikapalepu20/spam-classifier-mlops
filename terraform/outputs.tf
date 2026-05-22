output "server_public_ip" {

  value = aws_instance.deployment_server.public_ip

}

output "application_url" {

  value = "http://${aws_instance.deployment_server.public_ip}:5000"

}