// Initialize chart data
let chartData = [];
let signalData = null;

// Initialize chart
function initializeChart() {
    const canvas = document.getElementById('priceChart');
    if (!canvas) {
        console.error('Cannot find chart canvas element');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Cannot get canvas context');
        return;
    }
    
    console.log('Initializing chart...');
    
    // Advanced candlestick chart with technical analysis annotations
    window.priceChart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            labels: [], // Will be set by updateChart
            datasets: [{
                label: 'قیمت',
                data: [],
                color: {
                    up: 'rgba(0, 200, 83, 1)',
                    down: 'rgba(255, 61, 0, 1)',
                    unchanged: 'rgba(100, 100, 100, 1)',
                },
                borderColor: {
                    up: 'rgba(0, 200, 83, 1)',
                    down: 'rgba(255, 61, 0, 1)',
                    unchanged: 'rgba(100, 100, 100, 1)',
                },
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'زمان',
                        color: 'rgba(200, 200, 200, 1)'
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 15,
                        color: 'rgba(200, 200, 200, 1)'
                    },
                    grid: {
                        color: 'rgba(50, 50, 50, 0.5)'
                    }
                },
                y: {
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'قیمت',
                        color: 'rgba(200, 200, 200, 1)'
                    },
                    ticks: {
                        color: 'rgba(200, 200, 200, 1)'
                    },
                    grid: {
                        color: 'rgba(50, 50, 50, 0.5)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const dataPoint = context.raw;
                            if (!dataPoint) return '';
                            
                            // Format candlestick values
                            if (dataPoint.o !== undefined) {
                                return [
                                    `Open: ${dataPoint.o.toFixed(8)}`,
                                    `High: ${dataPoint.h.toFixed(8)}`,
                                    `Low: ${dataPoint.l.toFixed(8)}`,
                                    `Close: ${dataPoint.c.toFixed(8)}`
                                ];
                            }
                            return context.dataset.label + ': ' + context.formattedValue;
                        }
                    }
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgba(200, 200, 200, 1)'
                    }
                },
                // Add custom annotation for technical analysis
                annotation: {
                    annotations: {
                        // Will be added dynamically in updateChart
                    }
                }
            }
        }
    });
    
    // Add download chart button
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'btn btn-sm btn-outline-info mt-2';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> دانلود نمودار';
        downloadBtn.onclick = downloadChart;
        chartContainer.appendChild(downloadBtn);
    }
    
    console.log('Chart initialized with download button');
}

// Function to download chart as image
function downloadChart() {
    const canvas = document.getElementById('priceChart');
    if (!canvas) return;
    
    // Create a temporary link
    const link = document.createElement('a');
    link.download = 'chart-analysis.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
}

// Update chart with new data
function updateChart(candles, signal) {
    try {
        if (!window.priceChart) {
            console.error('Chart not initialized');
            return;
        }
        
        // Format candle data for chart
        const timestamps = [];
        const candlestickData = [];
        
        candles.forEach(candle => {
            // If timestamp is already a number, use it directly
            const timestamp = typeof candle.timestamp === 'number' 
                ? candle.timestamp 
                : new Date(candle.timestamp * 1000).getTime(); // Convert UNIX timestamp if needed
            
            const time = new Date(timestamp).toLocaleTimeString();
            timestamps.push(time);
            
            // Prepare candlestick data (OHLC format)
            candlestickData.push({
                o: parseFloat(candle.open),
                h: parseFloat(candle.high),
                l: parseFloat(candle.low),
                c: parseFloat(candle.close),
                t: time
            });
        });
        
        console.log('Candlestick data prepared:', candlestickData.length);
        
        // Set chart data for candlestick
        window.priceChart.data.labels = timestamps;
        window.priceChart.data.datasets[0] = {
            label: 'نمودار قیمت',
            data: candlestickData,
            color: {
                up: 'rgba(0, 200, 83, 1)',
                down: 'rgba(255, 61, 0, 1)',
                unchanged: 'rgba(100, 100, 100, 1)',
            },
            borderColor: {
                up: 'rgba(0, 200, 83, 1)',
                down: 'rgba(255, 61, 0, 1)',
                unchanged: 'rgba(100, 100, 100, 1)',
            },
            borderWidth: 2
        };
        
        // If we have a signal, add technical analysis levels and annotations
        if (signal && signal.signal !== 'NO_SIGNAL' && signal.signal !== 'ERROR') {
            // Create horizontal line datasets for key levels
            const entryLine = {
                type: 'line',
                label: 'نقطه ورود',
                yMin: signal.entry,
                yMax: signal.entry,
                borderColor: 'rgba(255, 255, 0, 0.7)',
                borderWidth: 2,
                borderDash: [5, 5]
            };
            
            const stopLossLine = {
                type: 'line',
                label: 'حد ضرر',
                yMin: signal.stop_loss,
                yMax: signal.stop_loss,
                borderColor: 'rgba(255, 0, 0, 0.7)',
                borderWidth: 2,
                borderDash: [5, 5]
            };
            
            const takeProfitLine = {
                type: 'line',
                label: 'حد سود',
                yMin: signal.take_profit,
                yMax: signal.take_profit,
                borderColor: 'rgba(0, 255, 0, 0.7)',
                borderWidth: 2,
                borderDash: [5, 5]
            };
            
            // Add annotations for technical analysis
            if (window.priceChart.options.plugins && window.priceChart.options.plugins.annotation) {
                window.priceChart.options.plugins.annotation.annotations = {
                    entryLine,
                    stopLossLine,
                    takeProfitLine
                };
                
                // Add trend lines based on signal type
                if (signal.signal === 'LONG') {
                    window.priceChart.options.plugins.annotation.annotations.trendLine = {
                        type: 'line',
                        label: {
                            display: true,
                            content: 'روند صعودی',
                            backgroundColor: 'rgba(0, 200, 83, 0.7)'
                        },
                        xMin: 0,
                        xMax: timestamps.length - 1,
                        yMin: Math.min(...candlestickData.map(c => c.l)) * 0.99,
                        yMax: signal.take_profit,
                        borderColor: 'rgba(0, 200, 83, 0.5)',
                        borderWidth: 2
                    };
                } else if (signal.signal === 'SHORT') {
                    window.priceChart.options.plugins.annotation.annotations.trendLine = {
                        type: 'line',
                        label: {
                            display: true,
                            content: 'روند نزولی',
                            backgroundColor: 'rgba(255, 61, 0, 0.7)'
                        },
                        xMin: 0,
                        xMax: timestamps.length - 1,
                        yMin: signal.take_profit,
                        yMax: Math.max(...candlestickData.map(c => c.h)) * 1.01,
                        borderColor: 'rgba(255, 61, 0, 0.5)',
                        borderWidth: 2
                    };
                }
            }
            
            // Add technical overlay dataset (e.g., moving average)
            const prices = candlestickData.map(c => c.c);
            const maValues = calculateSMA(prices, 20); // 20-period moving average
            
            window.priceChart.data.datasets.push({
                label: 'میانگین متحرک',
                data: maValues,
                fill: false,
                borderColor: 'rgba(75, 192, 255, 1)',
                borderWidth: 1.5,
                pointRadius: 0,
                type: 'line'
            });
        }
        
        // Update chart
        window.priceChart.update();
        
        // Display key levels on chart
        highlightKeyLevels(signal);
    } catch (e) {
        console.error('Error updating chart:', e);
    }
}

// Calculate Simple Moving Average
function calculateSMA(prices, period) {
    const result = [];
    
    // Fill with null until we have enough data points
    for (let i = 0; i < period - 1; i++) {
        result.push(null);
    }
    
    // Calculate SMA values
    for (let i = period - 1; i < prices.length; i++) {
        const sum = prices.slice(i - period + 1, i + 1).reduce((total, price) => total + price, 0);
        result.push(sum / period);
    }
    
    return result;
}

// Highlight key price levels on chart
function highlightKeyLevels(signal) {
    if (!signal || !signal.signal || signal.signal === 'NO_SIGNAL' || signal.signal === 'ERROR') {
        return;
    }
    
    try {
        const chartContainer = document.querySelector('.chart-container');
        if (!chartContainer) return;
        
        // Create levels info display
        const levelsInfo = document.createElement('div');
        levelsInfo.className = 'key-levels-info mt-3 p-2 border rounded bg-dark text-light';
        levelsInfo.innerHTML = `
            <h6 class="mb-2">سطوح کلیدی:</h6>
            <div class="d-flex justify-content-between mb-1">
                <span>ورود:</span>
                <span class="badge bg-warning text-dark">${signal.entry}</span>
            </div>
            <div class="d-flex justify-content-between mb-1">
                <span>حد ضرر:</span> 
                <span class="badge bg-danger">${signal.stop_loss}</span>
            </div>
            <div class="d-flex justify-content-between">
                <span>حد سود:</span>
                <span class="badge bg-success">${signal.take_profit}</span>
            </div>
        `;
        
        // Check if levels info already exists
        const existingLevelsInfo = chartContainer.querySelector('.key-levels-info');
        if (existingLevelsInfo) {
            chartContainer.replaceChild(levelsInfo, existingLevelsInfo);
        } else {
            chartContainer.appendChild(levelsInfo);
        }
    } catch (err) {
        console.error('Error adding key levels info:', err);
    }
}

// Function to load candle data from the results page
function loadChartData() {
    const candlesElement = document.getElementById('candles-data');
    if (!candlesElement) {
        console.error('Candles data element not found');
        return;
    }
    
    const candlesStr = candlesElement.value;
    if (!candlesStr) {
        console.error('Candles data is empty');
        return;
    }
    
    try {
        console.log('Raw candles data:', candlesStr);
        
        // Parse candle data
        const candles = JSON.parse(candlesStr);
        console.log('Parsed candles:', candles);
        
        // Check if candles data is valid
        if (!Array.isArray(candles) || candles.length === 0) {
            console.error('Invalid candles data format:', candles);
            return;
        }
        
        // Parse signal data
        const signalElement = document.getElementById('signal-data');
        if (signalElement) {
            const signalStr = signalElement.value;
            if (signalStr) {
                try {
                    signalData = JSON.parse(signalStr);
                    console.log('Signal data:', signalData);
                } catch (sigError) {
                    console.error('Error parsing signal data:', sigError);
                }
            }
        }
        
        // Initialize chart
        initializeChart();
        
        // Update chart with data
        updateChart(candles, signalData);
    } catch (e) {
        console.error('Error loading chart data:', e);
        console.error('Data that caused error:', candlesStr);
    }
}

// Load symbols when exchange is selected
function loadSymbols() {
    const exchange = document.getElementById('exchange').value;
    const symbolSelect = document.getElementById('symbol');
    
    // Clear existing options
    symbolSelect.innerHTML = '<option value="">Loading symbols...</option>';
    
    // Fetch symbols
    fetch('/get_symbols', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'exchange': exchange
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update symbol options
            symbolSelect.innerHTML = '<option value="">Select Symbol</option>';
            data.symbols.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                symbolSelect.appendChild(option);
            });
        } else {
            symbolSelect.innerHTML = '<option value="">Error loading symbols</option>';
            console.error('Error loading symbols:', data.error);
        }
    })
    .catch(error => {
        symbolSelect.innerHTML = '<option value="">Error loading symbols</option>';
        console.error('Error:', error);
    });
}

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart if on results page
    if (document.getElementById('candles-data')) {
        loadChartData();
    }
    
    // Add event listener for exchange select if on index page
    const exchangeSelect = document.getElementById('exchange');
    if (exchangeSelect) {
        exchangeSelect.addEventListener('change', loadSymbols);
    }
});
