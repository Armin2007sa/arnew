// Simple script for the results page that no longer uses charts
document.addEventListener('DOMContentLoaded', function() {
    console.log('Results page loaded');
    
    // Get signal data from hidden input
    const signalDataInput = document.getElementById('signal-data');
    
    if (signalDataInput) {
        try {
            // Parse JSON data
            const signalData = JSON.parse(signalDataInput.value);
            console.log("Signal data loaded:", signalData);
        } catch (error) {
            console.error("Error parsing signal data:", error);
        }
    }
});