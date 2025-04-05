function formatCurrency(value) {
    return '$' + parseFloat(value).toFixed(2);
}

function updateProfitDisplay(element, value) {
    const profitElement = document.getElementById(element);
    if (profitElement) {
        const profitValue = parseFloat(value);
        profitElement.textContent = formatCurrency(profitValue);
        
        if (profitValue >= 0) {
            profitElement.classList.add('text-success');
            profitElement.classList.remove('text-danger');
        } else {
            profitElement.classList.add('text-danger');
            profitElement.classList.remove('text-success');
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Format profit/loss values
    const totalProfitElement = document.getElementById('total-profit');
    if (totalProfitElement) {
        updateProfitDisplay('total-profit', totalProfitElement.textContent);
    }
});