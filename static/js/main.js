// Global variables
let messages = [];
let isAnalyzing = false;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            const activeTab = document.querySelector('.tab-pane.active');
            if (activeTab.id === 'single') {
                analyzeSingle();
            } else {
                analyzeConversation();
            }
        }
    });

    // Initialize message input handlers
    initializeMessageInputs();
});

function initializeMessageInputs() {
    const newMessageInput = document.getElementById('newMessage');
    if (newMessageInput) {
        newMessageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                addMessage();
            }
        });
    }
}

// Loading State Management
function showLoadingState() {
    document.getElementById('loadingOverlay').classList.add('show');
    document.querySelectorAll('button').forEach(btn => btn.disabled = true);
}

function hideLoadingState() {
    document.getElementById('loadingOverlay').classList.remove('show');
    document.querySelectorAll('button').forEach(btn => btn.disabled = false);
}

// Single Message Analysis
async function analyzeSingle() {
    const messageInput = document.getElementById('singleMessage');
    const message = messageInput.value.trim();
    
    if (!message) {
        showToast('Please enter a message to analyze', 'error');
        messageInput.focus();
        return;
    }

    if (isAnalyzing) return;
    isAnalyzing = true;

    try {
        showLoadingState();
        
        const response = await fetch('/analyze_message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displaySingleResult(result);
        showToast('Analysis completed', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message, 'error');
        document.getElementById('singleResult').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i> Analysis failed: ${error.message}
            </div>
        `;
    } finally {
        isAnalyzing = false;
        hideLoadingState();
    }
}

function displaySingleResult(result) {
    const resultDiv = document.getElementById('singleResult');
    
    let html = `
        <div class="result-box animate-fade-in">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="mb-0">
                    ${result.prediction === 'Red Flag' ? '🚩' : '✅'} 
                    ${result.prediction}
                </h5>
                <span class="badge severity-${result.severity.toLowerCase()}">
                    ${getSeverityIcon(result.severity)} ${result.severity}
                </span>
            </div>
            
            <div class="progress mb-3">
                <div class="progress-bar ${getConfidenceClass(result.confidence)}" 
                     role="progressbar" 
                     style="width: ${result.confidence}%" 
                     aria-valuenow="${result.confidence}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                    ${result.confidence.toFixed(1)}%
                </div>
            </div>
            
            <div class="context-box context-${result.severity.toLowerCase()}">
                <strong>${getContextIcon(result.severity)}</strong> 
                ${result.reasons[0]}
            </div>
            
            <div class="analysis-section mt-3">
                <h6>📋 Analysis:</h6>
                ${result.reasons.slice(1).map(reason => 
                    `<div class="analysis-item">${reason}</div>`
                ).join('')}
            </div>
    `;
    
    if (result.pattern_analysis.toxic_patterns) {
        html += `
            <div class="pattern-section mt-3">
                <h6>🔍 Detected Patterns:</h6>
                ${Object.entries(result.pattern_analysis.toxic_patterns).map(([category, patterns]) => `
                    <div class="pattern-card">
                        <strong>${formatCategory(category)}:</strong>
                        <ul class="mb-0">
                            ${patterns.map(pattern => `<li>${pattern}</li>`).join('')}
                        </ul>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    if (result.pattern_analysis.positive_patterns) {
        html += `
            <div class="positive-section mt-3">
                <h6>💚 Positive Patterns:</h6>
                ${Object.entries(result.pattern_analysis.positive_patterns).map(([category, patterns]) => `
                    <div class="pattern-card bg-light">
                        <strong>${formatCategory(category)}:</strong>
                        <ul class="mb-0">
                            ${patterns.map(pattern => `<li>${pattern}</li>`).join('')}
                        </ul>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    html += `
            <div class="actions mt-3">
                <button onclick="copyResult()" class="btn btn-sm btn-secondary">
                    <i class="fas fa-copy"></i> Copy Results
                </button>
                <button onclick="downloadReport()" class="btn btn-sm btn-primary">
                    <i class="fas fa-download"></i> Download Report
                </button>
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = html;
}

// Conversation Management
function addMessage() {
    const messageInput = document.getElementById('newMessage');
    const message = messageInput.value.trim();
    
    if (!message) {
        showToast('Please enter a message', 'error');
        return;
    }
    
    messages.push(message);
    updateMessageList();
    messageInput.value = '';
    messageInput.focus();
    
    showToast('Message added', 'success');
}

function updateMessageList() {
    const messageList = document.getElementById('messageList');
    messageList.innerHTML = messages.map((msg, idx) => `
        <div class="message-item animate-fade-in">
            <div class="d-flex justify-content-between align-items-center">
                <div class="message-content">
                    <span class="message-number">#${idx + 1}</span>
                    <p class="mb-0">${msg}</p>
                </div>
                <div class="message-actions">
                    <button onclick="editMessage(${idx})" class="btn btn-sm btn-outline-primary">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button onclick="removeMessage(${idx})" class="btn btn-sm btn-outline-danger">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `).join('');
}

async function analyzeConversation() {
    if (messages.length === 0) {
        showToast('Please add some messages first', 'error');
        return;
    }

    if (isAnalyzing) return;
    isAnalyzing = true;

    try {
        showLoadingState();
        
        const response = await fetch('/analyze_conversation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: messages })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayConversationResults(result);
        showToast('Conversation analysis completed', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message, 'error');
        document.getElementById('conversationResult').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i> Analysis failed: ${error.message}
            </div>
        `;
    } finally {
        isAnalyzing = false;
        hideLoadingState();
    }
}

// Utility Functions
function getSeverityIcon(severity) {
    switch(severity.toLowerCase()) {
        case 'critical': return '⛔';
        case 'high': return '⚠️';
        case 'medium': return '⚡';
        case 'low': return 'ℹ️';
        default: return '📝';
    }
}

function getContextIcon(severity) {
    switch(severity.toLowerCase()) {
        case 'critical': return '⛔ CRITICAL:';
        case 'high': return '⚠️ WARNING:';
        case 'medium': return '⚡ CAUTION:';
        case 'low': return 'ℹ️ INFO:';
        default: return '📝 NOTE:';
    }
}

function getConfidenceClass(confidence) {
    if (confidence >= 90) return 'bg-danger';
    if (confidence >= 70) return 'bg-warning';
    return 'bg-success';
}

function formatCategory(category) {
    return category.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.innerHTML = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }, 100);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
    return container;
}

function copyResult() {
    const resultText = document.getElementById('singleResult').innerText;
    navigator.clipboard.writeText(resultText)
        .then(() => showToast('Results copied to clipboard', 'success'))
        .catch(() => showToast('Failed to copy results', 'error'));
}

function downloadReport() {
    const result = document.getElementById('singleResult').innerText;
    const blob = new Blob([result], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `red-flag-analysis-${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('Report downloaded successfully', 'success');
}

function clearMessages() {
    if (messages.length === 0) {
        showToast('No messages to clear', 'info');
        return;
    }
    
    if (confirm('Are you sure you want to clear all messages?')) {
        messages = [];
        updateMessageList();
        document.getElementById('conversationResult').innerHTML = '';
        showToast('All messages cleared', 'success');
    }
}

function editMessage(index) {
    const message = messages[index];
    const newMessage = prompt('Edit message:', message);
    
    if (newMessage !== null && newMessage.trim() !== '') {
        messages[index] = newMessage.trim();
        updateMessageList();
        showToast('Message updated', 'success');
    }
}

function removeMessage(index) {
    if (confirm('Are you sure you want to remove this message?')) {
        messages.splice(index, 1);
        updateMessageList();
        showToast('Message removed', 'info');
    }
}

function displayConversationResults(analysis) {
    const resultDiv = document.getElementById('conversationResult');
    
    let html = `
        <div class="result-box animate-fade-in">
            <h5 class="mb-3">Conversation Analysis</h5>
            
            <!-- Overall Assessment -->
            <div class="overall-assessment mb-4">
                <div class="row">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-label">Total Messages</div>
                            <div class="stat-value">${analysis.overall_assessment.total_messages}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card red">
                            <div class="stat-label">Red Flags</div>
                            <div class="stat-value">${analysis.overall_assessment.red_flags}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card green">
                            <div class="stat-label">Green Flags</div>
                            <div class="stat-value">${analysis.overall_assessment.green_flags}</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card severity-${analysis.overall_assessment.severity.toLowerCase()}">
                            <div class="stat-label">Overall Severity</div>
                            <div class="stat-value">${analysis.overall_assessment.severity}</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Individual Message Analysis -->
            <h6 class="mb-3">Message Analysis:</h6>
    `;

    // Add each message analysis
    analysis.messages.forEach((result, index) => {
        html += `
            <div class="message-analysis-item ${result.prediction === 'Red Flag' ? 'red-flag' : 'green-flag'} mb-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">Message ${index + 1}</h6>
                    <span class="badge severity-${result.severity.toLowerCase()}">
                        ${getSeverityIcon(result.severity)} ${result.severity}
                    </span>
                </div>
                
                <div class="message-text mb-2">
                    <strong>Text:</strong> ${result.text}
                </div>
                
                <div class="prediction-details mb-2">
                    <span class="me-3">
                        ${result.prediction === 'Red Flag' ? '🚩' : '✅'} 
                        ${result.prediction}
                    </span>
                    <span class="confidence">
                        Confidence: ${result.confidence.toFixed(1)}%
                    </span>
                </div>
        `;

        // Add pattern analysis if exists
        if (result.pattern_analysis) {
            if (Object.keys(result.pattern_analysis.toxic_patterns || {}).length > 0) {
                html += `
                    <div class="patterns-detected mt-2">
                        <strong>Concerning Patterns:</strong>
                        <ul class="mb-2">
                `;
                
                Object.entries(result.pattern_analysis.toxic_patterns).forEach(([category, patterns]) => {
                    html += `
                        <li>
                            <strong>${formatCategory(category)}:</strong> 
                            ${patterns.join(', ')}
                        </li>
                    `;
                });
                
                html += `</ul></div>`;
            }

            if (Object.keys(result.pattern_analysis.positive_patterns || {}).length > 0) {
                html += `
                    <div class="positive-patterns">
                        <strong>Positive Patterns:</strong>
                        <ul class="mb-0">
                `;
                
                Object.entries(result.pattern_analysis.positive_patterns).forEach(([category, patterns]) => {
                    html += `
                        <li>
                            <strong>${formatCategory(category)}:</strong> 
                            ${patterns.join(', ')}
                        </li>
                    `;
                });
                
                html += `</ul></div>`;
            }
        }

        html += `</div>`;  // Close message-analysis-item
    });

    // Add download button
    html += `
            <div class="actions mt-4">
                <button onclick="downloadConversationReport()" class="btn btn-primary">
                    <i class="fas fa-download"></i> Download Full Report
                </button>
            </div>
        </div>
    `;

    resultDiv.innerHTML = html;
}

// Add this helper function for downloading conversation reports
function downloadConversationReport() {
    const result = document.getElementById('conversationResult').innerText;
    const blob = new Blob([result], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-analysis-${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('Conversation report downloaded successfully', 'success');
}