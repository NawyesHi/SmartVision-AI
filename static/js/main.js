document.addEventListener('DOMContentLoaded', function () {
    const videoFeed = document.getElementById('videoFeed');
    const startButton = document.getElementById('startCamera');
    const stopButton = document.getElementById('stopCamera');
    const statusMessage = document.getElementById('statusMessage');

    function updateStatusMessage(message, isError = false) {
        statusMessage.textContent = message;
        statusMessage.style.color = isError ? 'red' : 'green';
    }

    async function startCamera() {
        try {
            // Start the camera through Flask backend
            const response = await fetch('/api/start_camera');
            const data = await response.json();
            
            if (data.status === 'success') {
                // Set video feed source to the Flask video stream
                videoFeed.src = '/api/video_feed';
                updateStatusMessage('Camera started with face recognition');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error starting camera:', error);
            updateStatusMessage(error.message, true);
        }
    }

    async function stopCamera() {
        try {
            // Stop the camera through Flask backend
            const response = await fetch('/api/stop_camera');
            const data = await response.json();
            
            if (data.status === 'success') {
                videoFeed.src = '';
                updateStatusMessage('Camera stopped');
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error stopping camera:', error);
            updateStatusMessage(error.message, true);
        }
    }

    // Event listeners
    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);

    // Check if camera is available on page load
    fetch('/api/check_camera')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                updateStatusMessage('Camera ready');
            } else {
                updateStatusMessage('Camera not available', true);
            }
        })
        .catch(error => {
            console.error('Error checking camera:', error);
            updateStatusMessage('Error checking camera', true);
        });
});
