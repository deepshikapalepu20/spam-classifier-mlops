output "jenkins_public_ip" {
  value       = aws_instance.jenkins_server.public_ip
  description = "Public IP of Jenkins server"
}

output "jenkins_url" {
  value = "http://${aws_instance.jenkins_server.public_ip}:9090"
}