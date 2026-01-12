/**
 * JRA競馬予想アプリ - グラフ表示機能
 * Chart.jsを使用したダークテーマ対応のグラフ描画
 */

// Chart.jsのデフォルト設定
Chart.defaults.color = '#a3a3a3';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 12;

// カラーパレット
const chartColors = {
    primary: '#3b82f6',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#06b6d4',
    purple: '#8b5cf6',
    pink: '#ec4899',
    gradient: {
        blue: ['#3b82f6', '#2563eb'],
        green: ['#10b981', '#059669'],
        purple: ['#8b5cf6', '#7c3aed'],
    }
};

/**
 * 直近の勝率推移グラフを作成
 * @param {string} canvasId - canvas要素のID
 * @param {Array} labels - ラベル配列
 * @param {Array} data - データ配列
 */
function createWinRateChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.3)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: '勝率',
                data: data,
                borderColor: chartColors.primary,
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: chartColors.primary,
                pointBorderColor: '#0a0a0a',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#111111',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    padding: 12,
                    titleColor: '#f5f5f5',
                    bodyColor: '#a3a3a3',
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

/**
 * レース距離別成績の円グラフを作成
 * @param {string} canvasId - canvas要素のID
 * @param {Array} labels - ラベル配列
 * @param {Array} data - データ配列
 */
function createDistributionChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    chartColors.primary,
                    chartColors.success,
                    chartColors.warning,
                    chartColors.purple,
                    chartColors.info,
                ],
                borderColor: '#0a0a0a',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        color: '#a3a3a3'
                    }
                },
                tooltip: {
                    backgroundColor: '#111111',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    padding: 12,
                    titleColor: '#f5f5f5',
                    bodyColor: '#a3a3a3',
                }
            },
            cutout: '65%',
        }
    });
}

/**
 * 馬券回収率の棒グラフを作成
 * @param {string} canvasId - canvas要素のID
 * @param {Array} labels - ラベル配列
 * @param {Array} data - データ配列
 */
function createRecoveryRateChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '回収率',
                data: data,
                backgroundColor: data.map(value =>
                    value >= 100 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                ),
                borderColor: data.map(value =>
                    value >= 100 ? chartColors.success : chartColors.danger
                ),
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#111111',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    padding: 12,
                    titleColor: '#f5f5f5',
                    bodyColor: '#a3a3a3',
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

/**
 * 予測精度の混合グラフを作成
 * @param {string} canvasId - canvas要素のID
 * @param {Object} chartData - グラフデータ
 */
function createAccuracyChart(canvasId, chartData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    type: 'line',
                    label: '的中率',
                    data: chartData.accuracy,
                    borderColor: chartColors.primary,
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y1',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: chartColors.primary,
                    pointBorderColor: '#0a0a0a',
                    pointBorderWidth: 2,
                },
                {
                    type: 'bar',
                    label: 'レース数',
                    data: chartData.raceCount,
                    backgroundColor: 'rgba(139, 92, 246, 0.6)',
                    borderColor: chartColors.purple,
                    borderWidth: 2,
                    borderRadius: 6,
                    yAxisID: 'y',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                        color: '#a3a3a3'
                    }
                },
                tooltip: {
                    backgroundColor: '#111111',
                    borderColor: 'rgba(255, 255, 255, 0.12)',
                    borderWidth: 1,
                    padding: 12,
                    titleColor: '#f5f5f5',
                    bodyColor: '#a3a3a3',
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'レース数',
                        color: '#a3a3a3'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '的中率 (%)',
                        color: '#a3a3a3'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// エクスポート
window.chartUtils = {
    createWinRateChart,
    createDistributionChart,
    createRecoveryRateChart,
    createAccuracyChart,
    colors: chartColors
};
