/**
 * Scraping Manager - Handles web-based scraping execution with real-time progress
 *
 * Uses Server-Sent Events (SSE) for real-time progress updates from the server.
 */
class ScrapingManager {
    constructor() {
        this.eventSource = null;
        this.currentTaskId = null;

        // DOM elements
        this.form = document.getElementById('scrapingForm');
        this.startBtn = document.getElementById('startScrapingBtn');
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
        this.statDates = document.getElementById('statDates');
        this.statRaces = document.getElementById('statRaces');
        this.statEntries = document.getElementById('statEntries');
        this.statResults = document.getElementById('statResults');

        this.bindEvents();
    }

    bindEvents() {
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.cancelScraping());
        }
    }

    async handleSubmit(e) {
        e.preventDefault();

        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;
        const weekendsOnly = document.getElementById('weekendsOnly').checked;

        if (!startDate || !endDate) {
            this.showError('開始日と終了日を入力してください');
            return;
        }

        await this.startScraping({
            start_date: startDate,
            end_date: endDate,
            weekends_only: weekendsOnly
        });
    }

    async startScraping(params) {
        try {
            this.setFormEnabled(false);
            this.hideAllCards();

            const response = await fetch('/scraping/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });

            const data = await response.json();

            if (data.success) {
                this.currentTaskId = data.task_id;
                this.showProgressCard();
                this.connectProgress(data.task_id);
            } else {
                this.showError(data.error || 'スクレイピングの開始に失敗しました');
                this.setFormEnabled(true);
            }
        } catch (error) {
            console.error('Error starting scraping:', error);
            this.showError('サーバーとの通信に失敗しました');
            this.setFormEnabled(true);
        }
    }

    resumeTask(taskId) {
        this.currentTaskId = taskId;
        this.setFormEnabled(false);
        this.showProgressCard();
        this.connectProgress(taskId);
    }

    connectProgress(taskId) {
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource(`/scraping/api/progress/${taskId}`);

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
            // Try to reconnect once
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
            const response = await fetch(`/scraping/api/status/${taskId}`);
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
        const totalDates = progress.total_dates || 0;
        const processedDates = progress.processed_dates || 0;
        this.statDates.textContent = `${processedDates}/${totalDates}`;
        this.statRaces.textContent = progress.races_completed || 0;
        this.statEntries.textContent = progress.entries_saved || 0;
        this.statResults.textContent = progress.results_saved || 0;

        // Update current operation
        const currentDate = progress.current_date || '';
        const currentRace = progress.current_race || '';

        if (currentRace) {
            this.currentOpText.textContent = `${currentDate} - ${currentRace}`;
        } else if (currentDate) {
            this.currentOpText.textContent = `${currentDate} を処理中...`;
        } else {
            this.currentOpText.textContent = '処理中...';
        }

        // Update status text
        if (data.status === 'running') {
            this.progressStatus.textContent = 'スクレイピング中...';
        }
    }

    handleComplete(data) {
        this.hideAllCards();

        if (data.status === 'completed') {
            this.showResultCard(data);
        } else if (data.status === 'cancelled') {
            this.showResultCard(data, true);
        } else if (data.status === 'failed') {
            this.showError(data.error || 'スクレイピングが失敗しました');
        }

        this.setFormEnabled(true);
        this.currentTaskId = null;
    }

    showResultCard(data, cancelled = false) {
        const stats = data.stats || {};

        // Update title
        const resultTitle = document.getElementById('resultTitle');
        if (cancelled) {
            resultTitle.innerHTML = '<i class="bi bi-slash-circle text-warning"></i> キャンセル';
        } else {
            resultTitle.innerHTML = '<i class="bi bi-check-circle text-success"></i> 完了';
        }

        // Update stats
        document.getElementById('resultDatesProcessed').textContent = stats.dates_processed || 0;
        document.getElementById('resultDatesNoRaces').textContent = stats.dates_no_races || 0;
        document.getElementById('resultDatesFailed').textContent = stats.dates_failed || 0;

        document.getElementById('resultRacesFound').textContent = stats.total_races_found || 0;
        document.getElementById('resultRacesCompleted').textContent = stats.races_completed || 0;
        document.getElementById('resultRacesFilteredTrack').textContent = stats.races_filtered_by_track || 0;
        document.getElementById('resultRacesFilteredClass').textContent = stats.races_filtered_by_class || 0;

        document.getElementById('resultEntriesSaved').textContent = stats.total_entries_saved || 0;
        document.getElementById('resultResultsSaved').textContent = stats.total_results_saved || 0;
        document.getElementById('resultPayoutsSaved').textContent = stats.total_payouts_saved || 0;

        // Duration
        if (stats.duration_hours) {
            document.getElementById('resultDuration').textContent = `${stats.duration_hours}時間`;
        } else if (stats.duration_seconds) {
            const minutes = Math.round(stats.duration_seconds / 60);
            document.getElementById('resultDuration').textContent = `${minutes}分`;
        } else {
            document.getElementById('resultDuration').textContent = '-';
        }

        this.resultCard.style.display = 'block';
    }

    async cancelScraping() {
        if (!this.currentTaskId) return;

        try {
            this.cancelBtn.disabled = true;
            this.cancelBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> キャンセル中...';

            const response = await fetch(`/scraping/api/cancel/${this.currentTaskId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (!data.success) {
                console.error('Cancel failed:', data.error);
            }
            // Progress update will handle the UI change
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
                this.startBtn.innerHTML = '<i class="bi bi-play-circle"></i> スクレイピング開始';
            } else {
                this.startBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 実行中...';
            }
        }

        const inputs = this.form ? this.form.querySelectorAll('input') : [];
        inputs.forEach(input => {
            input.disabled = !enabled;
        });
    }

    hideAllCards() {
        if (this.progressCard) this.progressCard.style.display = 'none';
        if (this.resultCard) this.resultCard.style.display = 'none';
        if (this.errorCard) this.errorCard.style.display = 'none';
    }

    showProgressCard() {
        this.hideAllCards();
        if (this.progressCard) {
            this.progressCard.style.display = 'block';

            // Reset progress
            this.progressBar.style.width = '0%';
            this.progressPercent.textContent = '0%';
            this.progressStatus.textContent = '準備中...';
            this.currentOpText.textContent = '初期化中...';
            this.statDates.textContent = '0/0';
            this.statRaces.textContent = '0';
            this.statEntries.textContent = '0';
            this.statResults.textContent = '0';

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
