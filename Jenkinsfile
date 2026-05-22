pipeline {
    agent any
 
    environment {
        DOCKER_IMAGE = "deepshikapalepu/spam-classifier"
        DOCKER_TAG   = "v${BUILD_NUMBER}"
    }
 
    stages {
 
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/deepshikapalepu20/spam-classifier-mlops.git'
            }
        }
 
        stage('Train Model') {
            steps {
                sh 'pip3 install scikit-learn pandas --quiet --break-system-packages'
                sh 'python3 train_model.py'
            }
        }
 
        stage('Test Application') {
            steps {
                sh 'pip3 install flask scikit-learn prometheus_client --quiet --break-system-packages'
                sh '''
                python3 -c "
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
result = model.predict(['Win free prize now!'])[0]
assert result == 1, 'Spam detection failed!'
result2 = model.predict(['Hi, how are you?'])[0]
assert result2 == 0, 'Ham detection failed!'
print('All tests passed!')
"
                '''
            }
        }
 
        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
                sh "docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:latest"
            }
        }
 
        stage('Push to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin"
                    sh "docker push ${DOCKER_IMAGE}:${DOCKER_TAG}"
                    sh "docker push ${DOCKER_IMAGE}:latest"
                }
            }
        }
    }
 
    post {
        success {
            echo "Pipeline SUCCESS! Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
        }
        failure {
            echo "Pipeline FAILED! Check the logs above."
        }
    }
}
