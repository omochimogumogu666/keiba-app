// Main JavaScript for JRA Horse Racing Prediction App

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });

    // Smooth scroll to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });
});

// Utility function to format numbers
function formatNumber(num, decimals = 0) {
    return num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Utility function to format currency (JPY)
function formatCurrency(amount) {
    return '¥' + formatNumber(amount);
}

// Utility function to format probability as percentage
function formatProbability(prob) {
    return (prob * 100).toFixed(1) + '%';
}

// Fetch prediction data (example API call)
async function fetchPredictions(raceId) {
    try {
        const response = await fetch(`/predictions/api/race/${raceId}`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching predictions:', error);
        return null;
    }
}

// Display loading spinner
function showLoading(element) {
    element.innerHTML = `
        <div class="spinner-container">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
}

// Show error message
function showError(element, message) {
    element.innerHTML = `
        <div class="alert alert-danger">
            <i class="bi bi-exclamation-triangle"></i> ${message}
        </div>
    `;
}

// Confidence score badge color
function getConfidenceBadgeClass(score) {
    if (score >= 0.7) return 'bg-success';
    if (score >= 0.5) return 'bg-warning text-dark';
    return 'bg-secondary';
}

// Export functions for use in other scripts
window.RaceApp = {
    formatNumber,
    formatCurrency,
    formatProbability,
    fetchPredictions,
    showLoading,
    showError,
    getConfidenceBadgeClass
};
