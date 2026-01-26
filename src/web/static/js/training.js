/**
 * Training Manager - Handles web-based ML model training with real-time progress
 *
 * Uses Server-Sent Events (SSE) for real-time progress updates from the server.
 */
class TrainingManager {
    constructor() {
        this.eventSource = null;
        this.currentTaskId = null;
        this.featureChart = null;

        // DOM elements
        this.form = document.getElementById('trainingForm');
        this.startBtn = document.getElementById('startTrainingBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        this.progressCard = document.getElementById('progressCard');
        this.resultCard = document.getElementById('resultCard');
        this.errorCard = document.getElementById('errorCard');

        // Progress elements
        this.progressBar = document.getElementById('progressBar');
        this.progressPercent = document.getElementById('progressPercent');
        this.progressStatus = document.getElementById('progressStatus');
        this.currentOpText = document.getElementById('currentOpText');

        // Stats elements
        this.statSamples = document.getElementById('statSamples');
        this.statFeatures = document.getElementById('statFeatures');
        this.statEpoch = document.getElementById('statEpoch');
        this.lossStats = document.getElementById('lossStats');
        this.statTrainLoss = document.getElementById('statTrainLoss');
        this.statValLoss = document.getElementById('statValLoss');

        // NN options
        this.nnOptions = document.getElementById('nnOptions');
        this.modelTypeSelect = document.getElementById('modelType');

        this.bindEvents();
    }

    bindEvents() {
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelTraining());
        }
        if (this.modelTypeSelect) {
            this.modelTypeSelect.addEventListener('change', () => this.toggleNNOptions());
        }
    }

    toggleNNOptions() {
        if (this.modelTypeSelect.value === 'neural_network') {
            this.nnOptions.style.display = 'block';
        } else {
            this.nnOptions.style.display = 'none';
        }
    }

    async handleSubmit(e) {
        e.preventDefault();

        const modelType = document.getElementById('modelType').value;
        const taskType = document.getElementById('taskType').value;
        const saveModel = document.getElementById('saveModel').checked;

        const params = {
            model_type: modelType,
            task_type: taskType,
            save_model: saveModel
        };

        // Add NN-specific params if neural network selected
        if (modelType === 'neural_network') {
            params.batch_size = parseInt(document.getElementById('batchSize').value) || 64;
            params.n_epochs = parseInt(document.getElementById('nEpochs').value) || 100;
            params.learning_rate = parseFloat(document.getElementById('learningRate').value) || 0.001;
        }

        await this.startTraining(params);
    }

    async startTraining(params) {
        try {
            this.setFormEnabled(false);
            this.hideAllCards();

            const response = await fetch('/training/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();

            if (data.success) {
                this.currentTaskId = data.task_id;
                this.showProgressCard(params.model_type === 'neural_network');
                this.connectProgress(data.task_id);
            } else {
                this.showError(data.error || '学習の開始に失敗しました');
                this.setFormEnabled(true);
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.showError('サーバーとの通信に失敗しました');
            this.setFormEnabled(true);
        }
    }

    resumeTask(taskId) {
        this.currentTaskId = taskId;
        this.setFormEnabled(false);
        this.showProgressCard(true); // Assume NN for resume
        this.connectProgress(taskId);
    }

    connectProgress(taskId) {
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource(`/training/api/progress/${taskId}`);

        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateProgress(data);
        };

        this.eventSource.addEventListener('complete', (event) => {
            const data = JSON.parse(event.data);
            this.handleComplete(data);
            this.disconnect();
        });

        this.eventSource.addEventListener('error', (event) => {
            if (event.data) {
                const data = JSON.parse(event.data);
                this.showError(data.error || 'タスクが見つかりません');
            }
            this.disconnect();
            this.setFormEnabled(true);
        });

        this.eventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            if (this.eventSource.readyState === EventSource.CLOSED) {
                setTimeout(() => {
                    if (this.currentTaskId) {
                        this.checkTaskStatus(this.currentTaskId);
                    }
                }, 2000);
            }
        };
    }

    async checkTaskStatus(taskId) {
        try {
            const response = await fetch(`/training/api/status/${taskId}`);
            const data = await response.json();

            if (data.success && data.task) {
                if (data.task.status === 'running') {
                    this.connectProgress(taskId);
                } else {
                    this.handleComplete(data.task);
                }
            } else {
                this.showError('タスクの状態を取得できませんでした');
                this.setFormEnabled(true);
            }
        } catch (error) {
            console.error('Error checking task status:', error);
            this.showError('サーバーとの通信に失敗しました');
            this.setFormEnabled(true);
        }
    }

    updateProgress(data) {
        const progress = data.progress || {};
        const percent = progress.percent_complete || 0;

        // Update progress bar
        this.progressBar.style.width = `${percent}%`;
        this.progressBar.setAttribute('aria-valuenow', percent);
        this.progressPercent.textContent = `${percent.toFixed(1)}%`;

        // Update stats
        this.statSamples.textContent = progress.samples_loaded || 0;
        this.statFeatures.textContent = progress.features_count || 0;

        // Update epoch for NN
        if (progress.total_epochs > 0) {
            this.statEpoch.textContent = `${progress.current_epoch || 0}/${progress.total_epochs}`;
            this.lossStats.style.display = 'flex';

            if (progress.train_loss !== null && progress.train_loss !== undefined) {
                this.statTrainLoss.textContent = progress.train_loss.toFixed(4);
            }
            if (progress.val_loss !== null && progress.val_loss !== undefined) {
                this.statValLoss.textContent = progress.val_loss.toFixed(4);
            }
        } else {
            this.statEpoch.textContent = '-';
        }

        // Update current operation text
        if (progress.phase_text) {
            this.currentOpText.textContent = progress.phase_text;
        }

        // Update status text
        if (data.status === 'running') {
            this.progressStatus.textContent = '学習中...';
        }
    }

    handleComplete(data) {
        this.hideAllCards();

        if (data.status === 'completed') {
            this.showResultCard(data);
        } else if (data.status === 'cancelled') {
            this.showResultCard(data, true);
        } else if (data.status === 'failed') {
            this.showError(data.error || '学習が失敗しました');
        }

        this.setFormEnabled(true);
        this.currentTaskId = null;
    }

    showResultCard(data, cancelled = false) {
        const result = data.result || {};
        const testMetrics = result.test_metrics || {};

        // Update title
        const resultTitle = document.getElementById('resultTitle');
        if (cancelled) {
            resultTitle.innerHTML = '<i class="bi bi-slash-circle text-warning"></i> キャンセル';
        } else {
            resultTitle.innerHTML = '<i class="bi bi-check-circle text-success"></i> 学習完了';
        }

        // Update data info
        document.getElementById('resultTrainSamples').textContent = result.train_samples || 0;
        document.getElementById('resultValSamples').textContent = result.val_samples || 0;
        document.getElementById('resultTestSamples').textContent = result.test_samples || 0;

        // Update model info
        document.getElementById('resultModelType').textContent = result.model_type || '-';
        document.getElementById('resultTaskType').textContent = result.task_type || '-';

        // Duration
        if (result.training_time_seconds) {
            const seconds = result.training_time_seconds;
            if (seconds >= 60) {
                const minutes = Math.round(seconds / 60);
                document.getElementById('resultDuration').textContent = `${minutes}分`;
            } else {
                document.getElementById('resultDuration').textContent = `${seconds.toFixed(1)}秒`;
            }
        } else {
            document.getElementById('resultDuration').textContent = '-';
        }

        // Model path
        document.getElementById('resultModelPath').textContent = result.model_path || '保存なし';

        // Metrics
        const metricsRow = document.getElementById('metricsRow');
        metricsRow.innerHTML = '';

        if (result.task_type === 'regression') {
            this.addMetricCard(metricsRow, 'RMSE', testMetrics.rmse);
            this.addMetricCard(metricsRow, 'MAE', testMetrics.mae);
            this.addMetricCard(metricsRow, 'R²', testMetrics.r2_score);
        } else {
            this.addMetricCard(metricsRow, 'Accuracy', testMetrics.accuracy);
            this.addMetricCard(metricsRow, 'F1 Score', testMetrics.f1_score);
            this.addMetricCard(metricsRow, 'Precision', testMetrics.precision);
        }

        // Feature importance chart
        if (result.feature_importance && result.feature_importance.length > 0) {
            this.renderFeatureImportance(result.feature_importance);
            document.getElementById('featureImportanceContainer').style.display = 'block';
        } else {
            document.getElementById('featureImportanceContainer').style.display = 'none';
        }

        this.resultCard.style.display = 'block';
    }

    addMetricCard(container, label, value) {
        const col = document.createElement('div');
        col.className = 'col-md-4 text-center mb-2';

        const valueText = (value !== null && value !== undefined)
            ? (typeof value === 'number' ? value.toFixed(4) : value)
            : '-';

        col.innerHTML = `
            <div class="border rounded p-2">
                <div class="h5 mb-0 text-primary">${valueText}</div>
                <small class="text-muted">${label}</small>
            </div>
        `;
        container.appendChild(col);
    }

    renderFeatureImportance(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart').getContext('2d');

        // Destroy existing chart if any
        if (this.featureChart) {
            this.featureChart.destroy();
        }

        const labels = featureImportance.map(f => f.feature || f.index);
        const values = featureImportance.map(f => f.importance);

        this.featureChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '重要度',
                    data: values,
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: {
                                size: 10
                            }
                        }
                    }
                }
            }
        });
    }

    async cancelTraining() {
        if (!this.currentTaskId) return;

        try {
            this.cancelBtn.disabled = true;
            this.cancelBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> キャンセル中...';

            const response = await fetch(`/training/api/cancel/${this.currentTaskId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (!data.success) {
                console.error('Cancel failed:', data.error);
            }
        } catch (error) {
            console.error('Error cancelling:', error);
            this.cancelBtn.disabled = false;
            this.cancelBtn.innerHTML = '<i class="bi bi-x-circle"></i> キャンセル';
        }
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    setFormEnabled(enabled) {
        if (this.startBtn) {
            this.startBtn.disabled = !enabled;
            if (enabled) {
                this.startBtn.innerHTML = '<i class="bi bi-play-circle"></i> 学習開始';
            } else {
                this.startBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 実行中...';
            }
        }

        const inputs = this.form ? this.form.querySelectorAll('input, select') : [];
        inputs.forEach(input => {
            input.disabled = !enabled;
        });
    }

    hideAllCards() {
        if (this.progressCard) this.progressCard.style.display = 'none';
        if (this.resultCard) this.resultCard.style.display = 'none';
        if (this.errorCard) this.errorCard.style.display = 'none';
    }

    showProgressCard(isNeuralNetwork = false) {
        this.hideAllCards();
        if (this.progressCard) {
            this.progressCard.style.display = 'block';

            // Reset progress
            this.progressBar.style.width = '0%';
            this.progressPercent.textContent = '0%';
            this.progressStatus.textContent = '準備中...';
            this.currentOpText.textContent = '初期化中...';
            this.statSamples.textContent = '0';
            this.statFeatures.textContent = '0';
            this.statEpoch.textContent = '-';

            // Show/hide loss stats based on model type
            if (isNeuralNetwork) {
                this.lossStats.style.display = 'flex';
                this.statTrainLoss.textContent = '-';
                this.statValLoss.textContent = '-';
            } else {
                this.lossStats.style.display = 'none';
            }

            // Reset cancel button
            this.cancelBtn.disabled = false;
            this.cancelBtn.innerHTML = '<i class="bi bi-x-circle"></i> キャンセル';
        }
    }

    showError(message) {
        this.hideAllCards();
        if (this.errorCard) {
            document.getElementById('errorMessage').textContent = message;
            this.errorCard.style.display = 'block';
        }
    }
}
