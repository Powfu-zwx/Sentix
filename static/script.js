// 全局变量
let experimentsData = [];

// DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    initAnalyzeSection();
    loadExperiments();
    loadSystemInfo();
});

// 导航功能
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);

            // 更新导航状态
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            // 显示对应section
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });

            // 如果切换到实验区域，加载或刷新数据
            if (targetId === 'experiments') {
                if (experimentsData.length === 0) {
                    loadExperiments();
                } else {
                    // 重新绘制图表以确保显示
                    drawPerformanceChart(experimentsData);
                }
            }
        });
    });
}

// 初始化分析区域
function initAnalyzeSection() {
    const inputText = document.getElementById('inputText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const exampleBtns = document.querySelectorAll('.example-btn');

    // 分析按钮
    analyzeBtn.addEventListener('click', analyzeSentiment);

    // 清空按钮
    clearBtn.addEventListener('click', function() {
        inputText.value = '';
        showEmptyState();
    });

    // 示例按钮
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            inputText.value = this.getAttribute('data-text');
            analyzeSentiment();
        });
    });

    // 回车键分析
    inputText.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeSentiment();
        }
    });
}

// 情感分析
async function analyzeSentiment() {
    const inputText = document.getElementById('inputText');
    const text = inputText.value.trim();

    if (!text) {
        alert('请输入文本');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data);
        } else {
            alert('分析失败: ' + data.error);
        }
    } catch (error) {
        alert('请求失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示结果
function displayResult(data) {
    const resultEmpty = document.getElementById('resultEmpty');
    const resultContent = document.getElementById('resultContent');
    const sentimentBadge = document.getElementById('sentimentBadge');
    const sentimentText = document.getElementById('sentimentText');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceProgress = document.getElementById('confidenceProgress');
    const probPositive = document.getElementById('probPositive');
    const probNegative = document.getElementById('probNegative');
    const probPositiveValue = document.getElementById('probPositiveValue');
    const probNegativeValue = document.getElementById('probNegativeValue');
    const textLength = document.getElementById('textLength');

    // 隐藏空状态，显示结果
    resultEmpty.style.display = 'none';
    resultContent.style.display = 'block';

    // 情感标签
    if (data.sentiment_en === 'positive') {
        sentimentBadge.className = 'sentiment-badge positive';
        sentimentBadge.innerHTML = '<i class="fas fa-smile"></i><span>积极情感 😊</span>';
    } else {
        sentimentBadge.className = 'sentiment-badge negative';
        sentimentBadge.innerHTML = '<i class="fas fa-frown"></i><span>消极情感 😔</span>';
    }

    // 置信度
    const confidence = (data.confidence * 100).toFixed(1);
    confidenceValue.textContent = confidence + '%';
    confidenceProgress.style.width = confidence + '%';

    // 概率
    const posProb = (data.prob_positive * 100).toFixed(1);
    const negProb = (data.prob_negative * 100).toFixed(1);
    
    probPositive.style.width = posProb + '%';
    probNegative.style.width = negProb + '%';
    probPositiveValue.textContent = posProb + '%';
    probNegativeValue.textContent = negProb + '%';

    // 文本长度
    textLength.textContent = data.text_length;

    // 滚动到结果
    document.getElementById('resultCard').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

// 显示空状态
function showEmptyState() {
    const resultEmpty = document.getElementById('resultEmpty');
    const resultContent = document.getElementById('resultContent');
    resultEmpty.style.display = 'block';
    resultContent.style.display = 'none';
}

// 加载实验结果
async function loadExperiments() {
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();

        if (data.success) {
            experimentsData = data.data;
            displayExperiments(data.data);
        } else {
            console.error('加载实验结果失败:', data.error);
        }
    } catch (error) {
        console.error('请求失败:', error);
    }
}

// 显示实验结果
function displayExperiments(experiments) {
    const container = document.getElementById('experimentsList');
    container.innerHTML = '';

    // 找出最佳模型
    let bestIdx = 0;
    let bestF1 = 0;
    experiments.forEach((exp, idx) => {
        if (exp.f1 > bestF1) {
            bestF1 = exp.f1;
            bestIdx = idx;
        }
    });

    experiments.forEach((exp, idx) => {
        const isBest = idx === bestIdx;
        const card = document.createElement('div');
        card.className = 'experiment-card' + (isBest ? ' best' : '');
        
        card.innerHTML = `
            <div class="experiment-header">
                <div class="model-name-exp">${exp.model}</div>
                ${isBest ? '<div class="best-badge"><i class="fas fa-crown"></i> 最佳</div>' : ''}
            </div>
            <div class="experiment-stats">
                <div class="stat-item">
                    <div class="stat-label-exp">学习率</div>
                    <div class="stat-value-exp">${exp.learning_rate.toExponential(0)}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label-exp">准确率</div>
                    <div class="stat-value-exp">${(exp.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label-exp">F1分数</div>
                    <div class="stat-value-exp">${(exp.f1 * 100).toFixed(2)}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label-exp">训练时间</div>
                    <div class="stat-value-exp">${exp.training_time.toFixed(0)}s</div>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });

    // 绘制图表
    drawPerformanceChart(experiments);
}

// 绘制性能对比图表
function drawPerformanceChart(experiments) {
    const canvas = document.getElementById('performanceChart');
    if (!canvas) {
        console.error('Canvas元素未找到');
        return;
    }
    
    const ctx = canvas.getContext('2d');

    // 准备数据
    const labels = experiments.map(exp => {
        const modelShort = exp.model.split('-')[0];
        const lr = exp.learning_rate.toExponential(0);
        return `${modelShort} (${lr})`;
    });

    const accuracyData = experiments.map(exp => parseFloat((exp.accuracy * 100).toFixed(2)));
    const f1Data = experiments.map(exp => parseFloat((exp.f1 * 100).toFixed(2)));

    // 销毁旧图表
    if (window.performanceChart instanceof Chart) {
        window.performanceChart.destroy();
    }

    // 创建新图表
    window.performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '准确率 (%)',
                    data: accuracyData,
                    backgroundColor: 'rgba(99, 102, 241, 0.8)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 2,
                    borderRadius: 8
                },
                {
                    label: 'F1分数 (%)',
                    data: f1Data,
                    backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    borderColor: 'rgba(139, 92, 246, 1)',
                    borderWidth: 2,
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 14,
                            weight: '600'
                        },
                        padding: 20,
                        usePointStyle: true
                    }
                },
                title: {
                    display: true,
                    text: '模型性能对比 - 准确率与F1分数',
                    font: {
                        size: 18,
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 93,
                    max: 97,
                    ticks: {
                        font: {
                            size: 12
                        },
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 10
                        },
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
    
    console.log('图表绘制完成');
}

// 加载系统信息
async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system');
        const data = await response.json();

        if (data.success) {
            document.getElementById('deviceInfo').textContent = data.device;
            document.getElementById('cudaInfo').textContent = data.cuda_available ? '是 ✓' : '否';
            document.getElementById('torchVersion').textContent = data.torch_version;
        }
    } catch (error) {
        console.error('加载系统信息失败:', error);
    }
}

// 显示/隐藏加载动画
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

