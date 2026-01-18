/**
 * 予測タスクマネージャー
 *
 * 予測タスクの開始、進捗監視、キャンセル機能を提供します。
 * SSEを使用してリアルタイムで進捗を表示します。
 */

class PredictionManager {
    constructor() {
        this.taskId = null;
        this.eventSource = null;

        this.initElements();
        this.bindEvents();
    }

    initElements() {
        // Form elements
        this.form = document.getElementById('predictionForm');
        this.startBtn = document.getElementById('startPredictionBtn');
        this.targetDateInput = document.getElementById('targetDate');
        this.modelTypeSelect = document.getElementById('modelType');
        this.skipScrapingCheck = document.getElementById('skipScraping');

        // Card elements
        this.progressCard = document.getElementById('progressCard');
        this.resultCard = document.getElementById('resultCard');
        this.errorCard = document.getElementById('errorCard');

        // Progress elements
        this.progressBar = document.getElementById('progressBar');
        this.progressStatus = document.getElementById('progressStatus');
        this.progressPercent = document.getElementById('progressPercent');
        this.currentOpText = document.getElementById('currentOpText');
        this.cancelBtn = document.getElementById('cancelBtn');

        // Stats elements
        this.statRacesFound = document.getElementById('statRacesFound');
        this.statRacesProcessed = document.getElementById('statRacesProcessed');
        this.statPredictions = document.getElementById('statPredictions');

        // Result elements
        this.resultRaces = document.getElementById('resultRaces');
        this.resultPredictions = document.getElementById('resultPredictions');
        this.resultModel = document.getElementById('resultModel');

        // Error elements
        this.errorMessage = document.getElementById('errorMessage');
    }

    bindEvents() {
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelTask());
        }
    }

    async handleSubmit(e) {
        e.preventDefault();

        // Collect form data
        const data = {
            date: this.targetDateInput.value,
            model_type: this.modelTypeSelect.value,
            skip_scraping: this.skipScrapingCheck.checked
        };

        // Start the prediction task
        await this.startPrediction(data);
    }

    async startPrediction(data) {
        try {
            // Disable form
            this.setFormDisabled(true);
            this.hideAllCards();
            this.showCard(this.progressCard);
            this.resetProgress();

            // Call API to start prediction
            const response = await fetch('/predictions/api/generate/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                this.taskId = result.task_id;
                this.connectProgressStream(this.taskId);
            } else {
                this.showError(result.error || '予測の開始に失敗しました');
            }
        } catch (error) {
            console.error('Error starting prediction:', error);
            this.showError('予測の開始中にエラーが発生しました');
        }
    }

    connectProgressStream(taskId) {
        // Close any existing connection
        if (this.eventSource) {
            this.eventSource.close();
        }

        // Connect to SSE endpoint
        this.eventSource = new EventSource(`/predictions/api/generate/progress/${taskId}`);

        this.eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateProgress(data);
        };

        this.eventSource.addEventListener('complete', (event) => {
            const data = JSON.parse(event.data);
            this.handleComplete(data);
            this.eventSource.close();
        });

        this.eventSource.addEventListener('error', (event) => {
            if (event.data) {
                const data = JSON.parse(event.data);
                this.showError(data.error || '接続エラーが発生しました');
            }
            this.eventSource.close();
            this.setFormDisabled(false);
        });

        this.eventSource.onerror = () => {
            // SSE connection error - might be normal disconnect
            if (this.eventSource.readyState === EventSource.CLOSED) {
                console.log('SSE connection closed');
            }
        };
    }

    updateProgress(data) {
        const progress = data.progress || {};

        // Update progress bar
        const percent = progress.percent_complete || 0;
        this.progressBar.style.width = `${percent}%`;
        this.progressBar.setAttribute('aria-valuenow', percent);
        this.progressPercent.textContent = `${Math.round(percent)}%`;

        // Update status text
        if (progress.phase_text) {
            this.progressStatus.textContent = progress.phase_text;
            this.currentOpText.textContent = progress.phase_text;
        }

        // Update current race
        if (progress.current_race) {
            this.currentOpText.textContent = `${progress.phase_text} - ${progress.current_race}`;
        }

        // Update stats
        this.statRacesFound.textContent = progress.races_found || 0;
        this.statRacesProcessed.textContent = progress.races_processed || 0;
        this.statPredictions.textContent = progress.predictions_generated || 0;
    }

    handleComplete(data) {
        this.eventSource = null;
        this.setFormDisabled(false);

        const status = data.status;
        const result = data.result;

        if (status === 'completed' && result) {
            // Show success result
            this.resultRaces.textContent = result.races_processed || 0;
            this.resultPredictions.textContent = result.predictions_generated || 0;
            this.resultModel.textContent = result.model_name || '-';

            this.hideCard(this.progressCard);
            this.showCard(this.resultCard);

            // Update result title based on predictions count
            const resultTitle = document.getElementById('resultTitle');
            if (result.predictions_generated > 0) {
                resultTitle.innerHTML = '<i class="bi bi-check-circle text-success"></i> 予想完了';
            } else {
                resultTitle.innerHTML = '<i class="bi bi-info-circle text-warning"></i> レースが見つかりませんでした';
            }
        } else if (status === 'cancelled') {
            this.hideCard(this.progressCard);
            this.showError('タスクがキャンセルされました');
        } else if (status === 'failed') {
            this.hideCard(this.progressCard);
            this.showError(data.error || '予測に失敗しました');
        } else {
            // No races or predictions
            this.hideCard(this.progressCard);
            const progress = data.progress || {};
            if (progress.phase === 'no_races') {
                this.showError('今日のレースが見つかりませんでした。レース開催日かどうかを確認してください。');
            } else if (progress.phase === 'no_predictions') {
                this.showError('予測を生成できませんでした。出馬表データがあるか確認してください。');
            } else {
                this.showCard(this.resultCard);
                this.resultRaces.textContent = '0';
                this.resultPredictions.textContent = '0';
                this.resultModel.textContent = '-';
            }
        }
    }

    async cancelTask() {
        if (!this.taskId) return;

        try {
            const response = await fetch(`/predictions/api/generate/cancel/${this.taskId}`, {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.cancelBtn.disabled = true;
                this.cancelBtn.innerHTML = '<i class="bi bi-hourglass"></i> キャンセル中...';
            } else {
                console.error('Cancel failed:', result.error);
            }
        } catch (error) {
            console.error('Error cancelling task:', error);
        }
    }

    resumeTask(taskId) {
        this.taskId = taskId;
        this.setFormDisabled(true);
        this.hideAllCards();
        this.showCard(this.progressCard);
        this.connectProgressStream(taskId);
    }

    resetProgress() {
        this.progressBar.style.width = '0%';
        this.progressBar.setAttribute('aria-valuenow', 0);
        this.progressPercent.textContent = '0%';
        this.progressStatus.textContent = '準備中...';
        this.currentOpText.textContent = '初期化中...';
        this.statRacesFound.textContent = '0';
        this.statRacesProcessed.textContent = '0';
        this.statPredictions.textContent = '0';
        this.cancelBtn.disabled = false;
        this.cancelBtn.innerHTML = '<i class="bi bi-x-circle"></i> キャンセル';
    }

    setFormDisabled(disabled) {
        if (this.startBtn) {
            this.startBtn.disabled = disabled;
            if (disabled) {
                this.startBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 処理中...';
            } else {
                this.startBtn.innerHTML = '<i class="bi bi-play-circle"></i> 予想を開始';
            }
        }
        if (this.targetDateInput) this.targetDateInput.disabled = disabled;
        if (this.modelTypeSelect) this.modelTypeSelect.disabled = disabled;
        if (this.skipScrapingCheck) this.skipScrapingCheck.disabled = disabled;
    }

    showCard(card) {
        if (card) card.style.display = 'block';
    }

    hideCard(card) {
        if (card) card.style.display = 'none';
    }

    hideAllCards() {
        this.hideCard(this.progressCard);
        this.hideCard(this.resultCard);
        this.hideCard(this.errorCard);
    }

    showError(message) {
        this.hideAllCards();
        if (this.errorMessage) {
            this.errorMessage.textContent = message;
        }
        this.showCard(this.errorCard);
        this.setFormDisabled(false);
    }
}
