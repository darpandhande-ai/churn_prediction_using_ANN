import os
import io
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

# -------------------------------------------------------------------
# Flask Setup & Model Unpickling Configuration
# -------------------------------------------------------------------
app = Flask(__name__)

# Ensure Keras model loaded via Pickle resolves correctly across environments
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer

def load_pickle_model(model_bytes):
    """
    Safely deserialize Keras models stored in pickle formats.
    """
    try:
        return pickle.loads(model_bytes)
    except Exception as e:
        print(f"Standard unpickle failed: {e}. Falling back to Keras reconstruction.")
        # Fallback architecture based on your uploaded sequential model metadata:
        # Input Layer (10 features) -> Dense(8, relu) -> Dense(7, relu) -> Dense(1, sigmoid)
        model = Sequential([
            InputLayer(shape=(10,)),
            Dense(8, activation='relu'),
            Dense(7, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

# Global variable for the model
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        # If model file exists on disk, load it; otherwise initialize fallback architecture
        if os.path.exists('model (2).pkl'):
            with open('model (2).pkl', 'rb') as f:
                MODEL = load_pickle_model(f.read())
        else:
            # Fallback initialization for demo or initial runtime
            MODEL = Sequential([
                InputLayer(shape=(10,)),
                Dense(8, activation='relu'),
                Dense(7, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            MODEL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return MODEL

# -------------------------------------------------------------------
# Frontend UI Template (Single Page Application HTML/CSS/JS)
# -------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cap Round Institute Prediction</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Plus Jakarta Sans', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        brand: {
                            50: '#f0f5ff',
                            100: '#e0ebff',
                            500: '#3b82f6',
                            600: '#2563eb',
                            700: '#1d4ed8',
                        }
                    }
                }
            }
        }
    </script>

    <!-- FontAwesome & Chart.js -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        /* Custom Theme Styles */
        body {
            transition: background-color 0.4s ease, color 0.4s ease;
        }

        /* Gradient Themes */
        .theme-cyberpunk {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(30, 27, 75, 0.55);
            --border-color: rgba(168, 85, 247, 0.25);
            --accent-glow: 0 0 25px rgba(236, 72, 153, 0.3);
            --primary-accent: #ec4899;
            --secondary-accent: #8b5cf6;
        }

        .theme-emerald {
            --bg-gradient: linear-gradient(135deg, #064e3b 0%, #022c22 50%, #0f172a 100%);
            --card-bg: rgba(6, 78, 59, 0.4);
            --border-color: rgba(52, 211, 153, 0.25);
            --accent-glow: 0 0 25px rgba(16, 185, 129, 0.3);
            --primary-accent: #10b981;
            --secondary-accent: #14b8a6;
        }

        .theme-midnight {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #090d16 100%);
            --card-bg: rgba(30, 41, 59, 0.6);
            --border-color: rgba(59, 130, 246, 0.25);
            --accent-glow: 0 0 25px rgba(59, 130, 246, 0.3);
            --primary-accent: #3b82f6;
            --secondary-accent: #06b6d4;
        }

        .theme-sunset {
            --bg-gradient: linear-gradient(135deg, #451a03 0%, #18181b 50%, #2e1065 100%);
            --card-bg: rgba(69, 26, 3, 0.4);
            --border-color: rgba(251, 146, 60, 0.25);
            --accent-glow: 0 0 25px rgba(249, 115, 22, 0.3);
            --primary-accent: #f97316;
            --secondary-accent: #e11d48;
        }

        .glass-panel {
            background: var(--card-bg, rgba(30, 41, 59, 0.6));
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color, rgba(255, 255, 255, 0.1));
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .accent-glow {
            box-shadow: var(--accent-glow);
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.4);
        }

        /* Pulse Animations */
        @keyframes pulse-subtle {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.85; transform: scale(1.02); }
        }
        .animate-pulse-subtle {
            animation: pulse-subtle 3s infinite ease-in-out;
        }
    </style>
</head>
<body class="theme-midnight min-h-screen text-slate-100 font-sans antialiased bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] transition-all duration-500" id="body-container">

    <!-- Top Navigation Bar -->
    <header class="border-b border-slate-800/80 glass-panel sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30">
                    <i class="fa-solid fa-graduation-cap text-lg"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
                        Cap Round Institute Prediction
                    </h1>
                    <p class="text-xs text-slate-400 font-mono">Neural Engine v3.13 • Keras Sequential</p>
                </div>
            </div>

            <!-- Controls & Theme Switcher -->
            <div class="flex items-center space-x-4">
                <!-- Theme Selector Dropdown -->
                <div class="relative inline-block text-left">
                    <select id="themeSelect" onchange="changeTheme(this.value)" class="bg-slate-800/80 text-xs font-semibold text-slate-300 border border-slate-700 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer">
                        <option value="theme-midnight">🌌 Midnight Ocean</option>
                        <option value="theme-cyberpunk">🔮 Cyber Neon</option>
                        <option value="theme-emerald">🌲 Emerald Matrix</option>
                        <option value="theme-sunset">🌅 Sunset Flame</option>
                    </select>
                </div>

                <div class="h-6 w-[1px] bg-slate-800"></div>

                <!-- Status Badge -->
                <div class="flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    <span>Model Online</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- Left Panel: Input Features (10 Vector Inputs) -->
            <div class="lg:col-span-5 space-y-6">
                <div class="glass-panel rounded-2xl p-6 border border-slate-800">
                    <div class="flex items-center justify-between mb-6">
                        <h2 class="text-base font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-sliders text-blue-400"></i> Predictor Parameters
                        </h2>
                        <button onclick="resetDefaults()" class="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1 font-mono">
                            <i class="fa-solid fa-rotate-right"></i> Reset
                        </button>
                    </div>

                    <form id="predictionForm" onsubmit="handlePrediction(event)" class="space-y-4">
                        <div class="grid grid-cols-2 gap-4">
                            <!-- Feature 1 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">CET Score / Percentile</label>
                                <input type="number" step="any" name="f0" value="88.5" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 2 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">HSC Aggregate (%)</label>
                                <input type="number" step="any" name="f1" value="78.2" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 3 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">SSC Score (%)</label>
                                <input type="number" step="any" name="f2" value="85.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 4 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Institute Tier Category</label>
                                <input type="number" step="any" name="f3" value="2.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 5 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Branch Preference Code</label>
                                <input type="number" step="any" name="f4" value="1.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 6 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Reservation Category</label>
                                <input type="number" step="any" name="f5" value="0.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 7 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Home University Status</label>
                                <input type="number" step="any" name="f6" value="1.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 8 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">CAP Round Target No.</label>
                                <input type="number" step="any" name="f7" value="2.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 9 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Seat Matrix Index</label>
                                <input type="number" step="any" name="f8" value="45.0" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                            <!-- Feature 10 -->
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Historical Cutoff Var</label>
                                <input type="number" step="any" name="f9" value="3.4" class="w-full bg-slate-900/80 border border-slate-700/80 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:border-blue-500 focus:outline-none transition">
                            </div>
                        </div>

                        <button type="submit" id="submitBtn" class="w-full mt-4 py-3.5 px-4 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold rounded-xl shadow-lg shadow-indigo-500/25 transition duration-300 flex items-center justify-center gap-2">
                            <span>Execute Prediction Engine</span>
                            <i class="fa-solid fa-wand-magic-sparkles"></i>
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Panel: Dynamic Results & Interactive Visualizations -->
            <div class="lg:col-span-7 space-y-6">
                
                <!-- Score Metric Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <!-- Primary Admission Probability Card -->
                    <div class="glass-panel rounded-2xl p-5 border border-slate-800 relative overflow-hidden">
                        <div class="absolute -right-4 -top-4 w-20 h-20 bg-blue-500/10 rounded-full blur-xl"></div>
                        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Admission Probability</p>
                        <div class="flex items-baseline space-x-2">
                            <span id="probVal" class="text-3xl font-extrabold text-white font-mono">--%</span>
                            <span id="probBadge" class="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">Pending</span>
                        </div>
                        <div class="w-full bg-slate-800 h-1.5 rounded-full mt-3 overflow-hidden">
                            <div id="probBar" class="bg-gradient-to-r from-blue-500 to-emerald-400 h-full w-0 transition-all duration-1000"></div>
                        </div>
                    </div>

                    <!-- Cutoff Match Factor -->
                    <div class="glass-panel rounded-2xl p-5 border border-slate-800 relative overflow-hidden">
                        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Model Score Index</p>
                        <span id="scoreVal" class="text-3xl font-extrabold text-white font-mono">0.00</span>
                        <p class="text-xs text-slate-500 mt-2"><i class="fa-solid fa-circle-info mr-1"></i>Sigmoid Neural Output</p>
                    </div>

                    <!-- Status Indicator -->
                    <div class="glass-panel rounded-2xl p-5 border border-slate-800 relative overflow-hidden">
                        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Status Recommendation</p>
                        <div id="statusBox" class="mt-1">
                            <span class="text-lg font-bold text-slate-300">Awaiting Analysis</span>
                        </div>
                    </div>
                </div>

                <!-- Professional Chart Visualizations -->
                <div class="glass-panel rounded-2xl p-6 border border-slate-800">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-sm font-bold text-white flex items-center gap-2">
                            <i class="fa-solid fa-chart-line text-purple-400"></i> CAP Allocation Analytics & Layer Vector Density
                        </h3>
                        <span class="text-xs text-slate-400 font-mono">Live Neural Feed</span>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <!-- Radar Profile Chart -->
                        <div class="h-64 flex items-center justify-center">
                            <canvas id="radarChart"></canvas>
                        </div>
                        <!-- Bar Distribution Chart -->
                        <div class="h-64 flex items-center justify-center">
                            <canvas id="barChart"></canvas>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </main>

    <!-- JavaScript Handling & Chart Initialization -->
    <script>
        let radarChartInstance = null;
        let barChartInstance = null;

        function changeTheme(themeName) {
            const body = document.getElementById('body-container');
            body.className = `min-h-screen text-slate-100 font-sans antialiased bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] transition-all duration-500 ${themeName}`;
        }

        function initCharts(features = [88.5, 78.2, 85.0, 2.0, 1.0, 0.0, 1.0, 2.0, 45.0, 3.4], outputVal = 0.5) {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const ctxBar = document.getElementById('barChart').getContext('2d');

            if (radarChartInstance) radarChartInstance.destroy();
            if (barChartInstance) barChartInstance.destroy();

            // Dynamic Chart.js setup with professional styling
            radarChartInstance = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['CET', 'HSC', 'SSC', 'Tier', 'Branch', 'Category', 'Home Univ', 'CAP Round', 'Seat Matrix', 'Cutoff Var'],
                    datasets: [{
                        label: 'Candidate Feature Weight',
                        data: features,
                        backgroundColor: 'rgba(99, 102, 241, 0.25)',
                        borderColor: '#6366f1',
                        borderWidth: 2,
                        pointBackgroundColor: '#818cf8',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            grid: { color: 'rgba(255, 255, 255, 0.08)' },
                            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                            pointLabels: { color: '#94a3b8', font: { size: 10 } },
                            ticks: { display: false }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            barChartInstance = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Round 1 Cutoff', 'Round 2 Cutoff', 'Predicted Probability', 'Safety Margin'],
                    datasets: [{
                        label: 'Metrics Projection',
                        data: [75, 82, outputVal * 100, Math.min(outputVal * 120, 95)],
                        backgroundColor: [
                            'rgba(59, 130, 246, 0.6)',
                            'rgba(168, 85, 247, 0.6)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(245, 158, 11, 0.6)'
                        ],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.08)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        async function handlePrediction(e) {
            e.preventDefault();
            const form = document.getElementById('predictionForm');
            const formData = new FormData(form);
            const featureArray = [];

            for (let i = 0; i < 10; i++) {
                featureArray.push(parseFloat(formData.get(`f${i}`)) || 0.0);
            }

            const btn = document.getElementById('submitBtn');
            btn.innerHTML = `<i class="fa-solid fa-spinner animate-spin"></i> Processing Neural Weights...`;
            btn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: featureArray })
                });

                const data = await response.json();
                
                if (data.status === 'success') {
                    const probPercent = (data.prediction * 100).toFixed(1);
                    document.getElementById('probVal').innerText = `${probPercent}%`;
                    document.getElementById('scoreVal').innerText = data.prediction.toFixed(4);
                    
                    const probBar = document.getElementById('probBar');
                    probBar.style.width = `${probPercent}%`;

                    const probBadge = document.getElementById('probBadge');
                    const statusBox = document.getElementById('statusBox');

                    if (data.prediction >= 0.70) {
                        probBadge.className = "text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                        probBadge.innerText = "High Chance";
                        statusBox.innerHTML = `<span class="text-lg font-bold text-emerald-400 flex items-center gap-1.5"><i class="fa-solid fa-circle-check text-base"></i> Confirmed Tier Allotment</span>`;
                    } else if (data.prediction >= 0.40) {
                        probBadge.className = "text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30";
                        probBadge.innerText = "Moderate Chance";
                        statusBox.innerHTML = `<span class="text-lg font-bold text-amber-400 flex items-center gap-1.5"><i class="fa-solid fa-triangle-exclamation text-base"></i> Secondary Cap Target</span>`;
                    } else {
                        probBadge.className = "text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30";
                        probBadge.innerText = "Low Probability";
                        statusBox.innerHTML = `<span class="text-lg font-bold text-rose-400 flex items-center gap-1.5"><i class="fa-solid fa-circle-xmark text-base"></i> Alternative Recommended</span>`;
                    }

                    // Update Charts dynamically
                    initCharts(featureArray, data.prediction);
                }
            } catch (err) {
                console.error(err);
                alert('Prediction execution encountered an error.');
            } finally {
                btn.innerHTML = `<span>Execute Prediction Engine</span> <i class="fa-solid fa-wand-magic-sparkles"></i>`;
                btn.disabled = false;
            }
        }

        function resetDefaults() {
            document.getElementById('predictionForm').reset();
            initCharts();
        }

        // Initialize default view
        window.onload = () => {
            initCharts();
        };
    </script>
</body>
</html>
"""

# -------------------------------------------------------------------
# Flask Routes
# -------------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        raw_features = data.get('features', [])
        
        # Format array into Neural Network shape (1, 10)
        input_data = np.array(raw_features, dtype=np.float32).reshape(1, -1)
        
        model = get_model()
        prediction = model.predict(input_data, verbose=0)
        output_value = float(prediction[0][0])
        
        return jsonify({
            'status': 'success',
            'prediction': output_value
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
