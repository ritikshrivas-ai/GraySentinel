// graysentinel/static/js/app.js
document.addEventListener('DOMContentLoaded', function() {
    const scanForm = document.getElementById('scanForm');
    const scanProgress = document.getElementById('scanProgress');
    const progressBar = document.querySelector('.progress-bar');
    const progressText = document.getElementById('progressText');
    const resultsCard = document.getElementById('resultsCard');
    const resultsBody = document.getElementById('resultsBody');
    
    if (scanForm) {
        scanForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const scanType = document.getElementById('scanType').value;
            const target = document.getElementById('target').value;
            
            scanProgress.classList.remove('d-none');
            resultsCard.classList.add('d-none');
            
            fetch('/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': '{{ csrf_token() }}'
                },
                body: JSON.stringify({
                    scan_type: scanType,
                    target: target
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.task_id) {
                    monitorScanProgress(data.task_id);
                } else {
                    showError('Failed to start scan');
                }
            })
            .catch(error => {
                showError('Error: ' + error.message);
            });
        });
    }
    
    function monitorScanProgress(taskId) {
        let progress = 0;
        const interval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 10, 95);
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `Scan in progress... ${Math.round(progress)}%`;
        }, 1000);
        
        function checkStatus() {
            fetch(`/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.state === 'SUCCESS') {
                        clearInterval(interval);
                        progressBar.style.width = '100%';
                        progressText.textContent = 'Scan completed!';
                        showScanResults(data.result);
                    } else if (data.state === 'FAILURE') {
                        clearInterval(interval);
                        showError('Scan failed: ' + data.result);
                    } else {
                        setTimeout(checkStatus, 2000);
                    }
                })
                .catch(error => {
                    clearInterval(interval);
                    showError('Error checking scan status: ' + error.message);
                });
        }
        
        setTimeout(checkStatus, 3000);
    }
    
    function showScanResults(results) {
        resultsCard.classList.remove('d-none');
        
        if (results.error) {
            resultsBody.innerHTML = `
                <div class="alert alert-danger">
                    ${results.error}
                </div>
            `;
            return;
        }
        
        let content = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card mb-3">
                        <div class="card-header">
                            <h6>Scan Summary</h6>
                        </div>
                        <div class="card-body">
                            <p><strong>Target:</strong> ${results.target}</p>
                            <p><strong>Type:</strong> ${results.scan_type}</p>
                            <p><strong>Duration:</strong> ${results.time_elapsed} seconds</p>
                            <p><strong>Risk Score:</strong>
                                <span class="badge 
                                    ${results.risk_score < 40 ? 'bg-success' : 
                                      results.risk_score < 70 ? 'bg-warning' : 'bg-danger'}">
                                    ${results.risk_score}/100
                                </span>
                            </p>
                            <a href="/report/${results.scan_id}" class="btn btn-sm btn-primary mt-2">
                                Download Full Report
                            </a>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6>Risk Visualization</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="riskChart" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        resultsBody.innerHTML = content;
        
        // Render risk chart
        const ctx = document.getElementById('riskChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [
                        results.risk_score < 40 ? 1 : 0,
                        (results.risk_score >= 40 && results.risk_score < 70) ? 1 : 0,
                        results.risk_score >= 70 ? 1 : 0
                    ],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: { enabled: false }
                }
            }
        });
    }
    
    function showError(message) {
        scanProgress.classList.add('d-none');
        alert(message);
    }
});
