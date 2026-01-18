/**
 * JRA競馬予想アプリ - グラフ表示機能
 * Chart.js Premium Racing Terminal テーマ
 * Deep Navy × Gold アクセント
 */

// Chart.jsのデフォルト設定
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.1)';
Chart.defaults.font.family = "'DM Sans', 'Hiragino Sans', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.font.weight = 400;

// Premium Racing Terminal カラーパレット
const chartColors = {
    // プライマリ - ゴールド/アンバー
    primary: '#f59e0b',
    primaryLight: 'rgba(245, 158, 11, 0.15)',
    primaryGlow: 'rgba(245, 158, 11, 0.4)',

    // セカンダリ - エメラルド
    secondary: '#10b981',
    secondaryLight: 'rgba(16, 185, 129, 0.15)',

    // セマンティック
    success: '#22c55e',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#3b82f6',
    muted: '#64748b',

    // 背景色
    bgBase: '#0a0e1a',
    bgElevated: '#111827',
    bgSubtle: '#1e293b',

    // テキスト
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    textMuted: '#64748b',

    // グラデーション用パレット - Premium Racing
    palette: [
        '#f59e0b',  // Gold (Primary)
        '#fbbf24',  // Amber Light
        '#10b981',  // Emerald
        '#3b82f6',  // Blue
        '#8b5cf6',  // Violet
        '#ec4899',  // Pink
        '#06b6d4',  // Cyan
    ]
};

// 共通ツールチップ設定
const tooltipConfig = {
    backgroundColor: '#1e293b',
    borderColor: 'rgba(148, 163, 184, 0.2)',
    borderWidth: 1,
    padding: 14,
    titleColor: '#f8fafc',
    titleFont: {
        family: "'Cormorant Garamond', serif",
        size: 14,
        weight: 600
    },
    bodyColor: '#94a3b8',
    bodyFont: {
        family: "'JetBrains Mono', monospace",
        size: 12,
        weight: 400
    },
    cornerRadius: 10,
    displayColors: true,
    boxPadding: 6,
    caretSize: 8,
    caretPadding: 10,
};

// 共通スケール設定
const scaleConfig = {
    y: {
        beginAtZero: true,
        grid: {
            color: 'rgba(148, 163, 184, 0.08)',
            drawBorder: false,
        },
        ticks: {
            color: '#64748b',
            font: {
                family: "'JetBrains Mono', monospace",
                size: 11
            },
            padding: 10,
        },
        border: {
            display: false
        }
    },
    x: {
        grid: {
            display: false
        },
        ticks: {
            color: '#64748b',
            font: {
                family: "'JetBrains Mono', monospace",
                size: 11
            },
            padding: 10,
        },
        border: {
            display: false
        }
    }
};

/**
 * レース数推移グラフ - ゴールドラインチャート
 * @param {string} canvasId - canvas要素のID
 * @param {Array} labels - ラベル配列
 * @param {Array} data - データ配列
 */
function createWinRateChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(245, 158, 11, 0.25)');
    gradient.addColorStop(0.5, 'rgba(245, 158, 11, 0.08)');
    gradient.addColorStop(1, 'rgba(245, 158, 11, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'レース数',
                data: data,
                borderColor: chartColors.primary,
                backgroundColor: gradient,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 8,
                pointBackgroundColor: chartColors.primary,
                pointBorderColor: chartColors.bgElevated,
                pointBorderWidth: 3,
                pointHoverBackgroundColor: chartColors.primary,
                pointHoverBorderColor: chartColors.textPrimary,
                pointHoverBorderWidth: 3,
            }]
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
                    display: false
                },
                tooltip: {
                    ...tooltipConfig,
                    callbacks: {
                        title: function(tooltipItems) {
                            return tooltipItems[0].label;
                        },
                        label: function(context) {
                            return ' ' + context.parsed.y + ' レース';
                        }
                    }
                }
            },
            scales: scaleConfig,
            animation: {
                duration: 1000,
                easing: 'easeOutQuart',
            }
        }
    });
}

/**
 * 競馬場分布ドーナツグラフ - Premium リング
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
                backgroundColor: chartColors.palette,
                borderColor: chartColors.bgElevated,
                borderWidth: 3,
                hoverBorderWidth: 3,
                hoverBorderColor: chartColors.textPrimary,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        color: chartColors.textSecondary,
                        font: {
                            family: "'DM Sans', sans-serif",
                            size: 12,
                            weight: 500
                        },
                        generateLabels: function(chart) {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, i) => {
                                    const value = data.datasets[0].data[i];
                                    return {
                                        text: `${label}: ${value}`,
                                        fillStyle: data.datasets[0].backgroundColor[i],
                                        strokeStyle: chartColors.bgElevated,
                                        lineWidth: 2,
                                        hidden: false,
                                        index: i
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    ...tooltipConfig,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return ` ${context.parsed} レース (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '65%',
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

/**
 * 馬券回収率バーグラフ - 収支表示
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
                    value >= 100 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
                ),
                borderColor: data.map(value =>
                    value >= 100 ? chartColors.success : chartColors.danger
                ),
                borderWidth: 2,
                borderRadius: 8,
                hoverBackgroundColor: data.map(value =>
                    value >= 100 ? chartColors.success : chartColors.danger
                ),
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
                    ...tooltipConfig,
                    callbacks: {
                        label: function(context) {
                            const rate = context.parsed.y.toFixed(1);
                            const status = context.parsed.y >= 100 ? '✓ プラス' : '△ マイナス';
                            return [` 回収率: ${rate}%`, ` ${status}`];
                        }
                    }
                }
            },
            scales: {
                ...scaleConfig,
                y: {
                    ...scaleConfig.y,
                    ticks: {
                        ...scaleConfig.y.ticks,
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            animation: {
                duration: 800,
                easing: 'easeOutQuart',
            }
        }
    });
}

/**
 * 予測精度の混合グラフ - デュアルアクシス
 * @param {string} canvasId - canvas要素のID
 * @param {Object} chartData - グラフデータ
 */
function createAccuracyChart(canvasId, chartData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const accuracyGradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
    accuracyGradient.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
    accuracyGradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    type: 'line',
                    label: '的中率',
                    data: chartData.accuracy,
                    borderColor: chartColors.secondary,
                    backgroundColor: accuracyGradient,
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y1',
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: chartColors.secondary,
                    pointBorderColor: chartColors.bgElevated,
                    pointBorderWidth: 3,
                },
                {
                    type: 'bar',
                    label: 'レース数',
                    data: chartData.raceCount,
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: chartColors.primary,
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
                        padding: 20,
                        usePointStyle: true,
                        color: chartColors.textSecondary,
                        font: {
                            family: "'DM Sans', sans-serif",
                            size: 12,
                            weight: 500
                        }
                    }
                },
                tooltip: tooltipConfig
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'レース数',
                        color: chartColors.textMuted,
                        font: {
                            family: "'DM Sans', sans-serif",
                            size: 11,
                            weight: 600
                        }
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.08)',
                        drawBorder: false,
                    },
                    ticks: {
                        color: chartColors.textMuted,
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 11
                        },
                        padding: 10,
                    },
                    border: { display: false }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: '的中率 (%)',
                        color: chartColors.textMuted,
                        font: {
                            family: "'DM Sans', sans-serif",
                            size: 11,
                            weight: 600
                        }
                    },
                    ticks: {
                        color: chartColors.textMuted,
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 11
                        },
                        padding: 10,
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    border: { display: false }
                },
                x: {
                    grid: { display: false },
                    ticks: {
                        color: chartColors.textMuted,
                        font: {
                            family: "'JetBrains Mono', monospace",
                            size: 11
                        },
                        padding: 10,
                    },
                    border: { display: false }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart',
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
