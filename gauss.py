from flask import Flask, request, jsonify, render_template
import numpy as np
from scipy.stats import norm, skew, kurtosis

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_gauss', methods=['POST'])
def generate_gauss():
    try:
        data = request.get_json()
        values = np.array(data['values'])
        usl = data.get('usl')
        lsl = data.get('lsl')
        
        # Convertir valores NumPy a tipos Python nativos
        mean = float(np.mean(values))
        stddev = float(np.std(values))
        range_val = float(np.ptp(values))
        
        # Calcular moda y convertir a float
        mode = float(np.unique(values, return_counts=True)[0][np.argmax(np.unique(values, return_counts=True)[1])])
        
        # Calcular estadísticas y convertir a float
        skewness = float(skew(values))
        kurt = float(kurtosis(values))

        # Calcular Cp y Cpk si se proporcionan límites
        if usl is not None and lsl is not None:
            cp = float(round((usl - lsl) / (6 * stddev), 2))
            cpk = float(round(min((usl - mean) / (3 * stddev), (mean - lsl) / (3 * stddev)), 2))
        else:
            cp = cpk = None
        
        # Calcular DPMO y yield
        defects = np.sum((values < lsl) | (values > usl))
        opportunities = len(values)
        dpmo = (defects / opportunities) * 1e6 if opportunities > 0 else None
        yield_val = (1 - (defects / opportunities)) * 100 if opportunities > 0 else None
        
        # Generar puntos para la distribución
        x = np.linspace(mean - 3 * stddev, mean + 3 * stddev, 100)
        y = norm.pdf(x, mean, stddev)
        y = y * (0.5 / max(y))

        # Generar distribución normal estándar
        x_standard = np.linspace(-3, 3, 100)
        y_standard = norm.pdf(x_standard, 0, 1)

        # Calcular límites de control
        control_limits = {
            '1_stddev': (float(mean - stddev), float(mean + stddev)),
            '2_stddev': (float(mean - 2 * stddev), float(mean + 2 * stddev)),
            '3_stddev': (float(mean - 3 * stddev), float(mean + 3 * stddev))
        }

        return jsonify({
            'mean': mean,
            'stddev': stddev,
            'range': range_val,
            'mode': mode,
            'skewness': skewness,
            'kurtosis': kurt,
            'cp': cp,
            'cpk': cpk,
            'dpmo': dpmo,
            'yield': yield_val,
            'x': x.tolist(), 
            'y': y.tolist(),
            'x_standard': x_standard.tolist(),
            'y_standard': y_standard.tolist(),
            'control_limits': control_limits
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)