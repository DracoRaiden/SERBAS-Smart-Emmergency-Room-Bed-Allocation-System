from flask import Flask, render_template, request, jsonify, session, send_from_directory
import sqlite3
import os
import re 
from datetime import datetime, timedelta
import joblib 
import numpy as np

# Create Flask app
app = Flask(__name__)
app.secret_key = 'serbas_hospital_secret_key' 

DB_NAME = 'hospital_hybrid_final.db'

# --- MODEL LOADING ---
ML_MODEL = None
KMEANS_MODEL = None
SCALER = None
HIGH_RISK_CLUSTER = None
MODEL_LOAD_SUCCESS = False
KMEANS_LOAD_SUCCESS = False

RF_MODEL_PATH = 'random_forest_model.joblib'
KMEANS_MODEL_PATH = 'kmeans_model.joblib'
SCALER_PATH = 'vitals_scaler.joblib'
HIGH_RISK_INDEX_PATH = 'high_risk_cluster_index.joblib'

def load_ml_models():
    global ML_MODEL, MODEL_LOAD_SUCCESS, KMEANS_MODEL, SCALER, HIGH_RISK_CLUSTER, KMEANS_LOAD_SUCCESS
    try:
        ML_MODEL = joblib.load(RF_MODEL_PATH)
        MODEL_LOAD_SUCCESS = True
        print(f"✅ Supervised Model Loaded")
    except Exception as e:
        print(f"❌ Supervised Failed: {e}")
        MODEL_LOAD_SUCCESS = False

    try:
        KMEANS_MODEL = joblib.load(KMEANS_MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        HIGH_RISK_CLUSTER = joblib.load(HIGH_RISK_INDEX_PATH)
        KMEANS_LOAD_SUCCESS = True
        print(f"✅ Unsupervised Loaded (Cluster {HIGH_RISK_CLUSTER})")
    except Exception as e:
        print(f"⚠️ Unsupervised Failed: {e}")
        KMEANS_LOAD_SUCCESS = False

load_ml_models()

# --- HELPER FUNCTIONS ---

def get_next_bed_id(cursor, hospital_id, prefix):
    """Finds the next available ID number for a bed type"""
    cursor.execute("SELECT id FROM beds WHERE hospital_id=? AND id LIKE ?", (hospital_id, f'{prefix}%'))
    existing_ids = [row[0] for row in cursor.fetchall()]
    max_num = 0
    for bid in existing_ids:
        match = re.search(r'\d+', bid)
        if match:
            num = int(match.group())
            if num > max_num: max_num = num
    return max_num + 1

def adjust_bed_capacity(conn, hospital_id, bed_type, target_count, prefix, name):
    c = conn.cursor()
    c.execute("SELECT id, status FROM beds WHERE hospital_id=? AND type=?", (hospital_id, bed_type))
    current_beds = c.fetchall()
    current_count = len(current_beds)
    
    if target_count > current_count:
        needed = target_count - current_count
        next_num = get_next_bed_id(c, hospital_id, prefix)
        for i in range(needed):
            new_id = f"{prefix}{next_num + i:03d}"
            c.execute("INSERT INTO beds VALUES (?, ?, ?, ?, 'available', NULL, NULL)", (new_id, hospital_id, bed_type, name))
            
    elif target_count < current_count:
        to_remove = current_count - target_count
        available_ids = [b[0] for b in current_beds if b[1] == 'available']
        if len(available_ids) < to_remove:
            raise Exception(f"Cannot reduce {bed_type.upper()} beds to {target_count}. {current_count - len(available_ids)} occupy beds.")
        ids_to_delete = available_ids[-to_remove:]
        for bid in ids_to_delete:
            c.execute("DELETE FROM beds WHERE id=?", (bid,))

def calculate_expected_discharge(admission_date, expected_stay_days):
    if admission_date and expected_stay_days:
        try:
            adm = datetime.strptime(admission_date, '%Y-%m-%d')
            dis = adm + timedelta(days=int(expected_stay_days))
            return dis.strftime('%Y-%m-%d')
        except: return None
    return None

def get_rule_based_severity(data):
    try:
        hr = int(data.get('heart_rate', 0))
        spo2 = int(data.get('spO2', 100))
        temp = float(data.get('temperature', 37.0))
        age = int(data.get('age', 30))
        bp_sys = int(data.get('blood_pressure_systolic', 120))
    except: return 'low'

    if spo2 < 92 or hr > 130 or temp >= 40.0 or bp_sys < 95: return 'high'
    elif (spo2 >= 92 and spo2 < 95) or (hr >= 100 and hr <= 130) or (temp >= 38.5 and temp < 40.0) or (age >= 70): return 'medium'
    return 'low'

def predict_severity_ml(data):
    rule = get_rule_based_severity(data)
    if not MODEL_LOAD_SUCCESS: return rule, f"ML Offline. Using Rule: {rule.upper()}", 0.5
    try:
        feats = np.array([float(data.get('age',0)), float(data.get('heart_rate',0)), float(data.get('blood_pressure_systolic',0)), float(data.get('blood_pressure_diastolic',0)), float(data.get('spO2',0)), float(data.get('temperature',0))]).reshape(1, -1)
        ml = ML_MODEL.predict(feats)[0].lower()
        msg = f"ML: {ml.upper()}. " + ("Differs from Rule." if ml != rule else "Matches Rule.")
        return ml, msg, 0.99
    except: return rule, "ML Error", 0.5

def run_unsupervised_model(data):
    if not KMEANS_LOAD_SUCCESS: return 0, 'Normal (Mock)'
    try:
        feats = np.array([float(data.get('heart_rate',0)), float(data.get('blood_pressure_systolic',0)), float(data.get('blood_pressure_diastolic',0)), float(data.get('spO2',0)), float(data.get('temperature',0))]).reshape(1, -1)
        scaled = SCALER.transform(feats)
        clust = KMEANS_MODEL.predict(scaled)[0]
        if clust == HIGH_RISK_CLUSTER: return 40, f'⚠️ High Risk Cluster ({clust})'
        return 0, f'Cluster {clust} (Normal)'
    except: return 0, 'Error'

def calculate_priority_score(sev, risk, doc, bonus):
    score = bonus
    if sev == 'high': score += 40
    elif sev == 'medium': score += 20
    if risk == 'critical': score += 40
    elif risk == 'moderate': score += 20
    if doc == 'icu': score += 20
    return score

def solve_bed_csp(cursor, hid, priority, sev, pref):
    mandatory = 'general'
    if sev == 'high': mandatory = 'icu'
    elif sev in ['medium', 'low'] and pref == 'flexible': mandatory = 'flexible'
    elif sev == 'low' and pref == 'flexible': mandatory = 'flexible'

    search_order = list(dict.fromkeys([mandatory, pref, 'icu' if sev == 'high' else 'general', 'general', 'flexible']))
    
    for bed_type in search_order:
        cursor.execute(f"SELECT id FROM beds WHERE hospital_id=? AND type=? AND status='available' LIMIT 1", (hid, bed_type))
        bed = cursor.fetchone()
        if bed: return bed[0], f"Allocated {bed[0]} ({bed_type.upper()}) based on {sev.upper()} severity."
    
    return None, f"No beds found for {mandatory.upper()}."

# --- DB INIT ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS hospitals (id TEXT PRIMARY KEY, name TEXT, address TEXT, contact TEXT, total_beds INT, icu_beds INT, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS patients (id TEXT PRIMARY KEY, name TEXT, age INT, heart_rate INT, bp_systolic INT, bp_diastolic INT, spO2 INT, temperature REAL, blood_group TEXT, condition TEXT, severity TEXT, health_risk TEXT, doctor_recommendation TEXT, priority_score INT, status TEXT, bed_id TEXT, admission_date TEXT, discharge_date TEXT, expected_stay_days INT, hospital_id TEXT, extended_stay INT DEFAULT 0, risk_flag TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS beds (id TEXT PRIMARY KEY, hospital_id TEXT, type TEXT, ward TEXT, status TEXT, patient_id TEXT, last_occupied_date TEXT)''')
    
    c.execute("SELECT COUNT(*) FROM hospitals")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO hospitals VALUES ('HOSP001', 'CMH Rawalpindi', 'Cantt', '051-111', 150, 20, 'password123')")
        for i in range(1, 101): c.execute("INSERT OR IGNORE INTO beds VALUES (?, ?, 'general', 'Ward A', 'available', NULL, NULL)", (f"BED{i:03d}", 'HOSP001'))
        for i in range(1, 21): c.execute("INSERT OR IGNORE INTO beds VALUES (?, ?, 'icu', 'ICU 1', 'available', NULL, NULL)", (f"ICU{i:03d}", 'HOSP001'))
        for i in range(1, 31): c.execute("INSERT OR IGNORE INTO beds VALUES (?, ?, 'flexible', 'Flex 1', 'available', NULL, NULL)", (f"FLEX{i:03d}", 'HOSP001'))
        create_sample_patients(conn, 'HOSP001')

    conn.commit()
    conn.close()

def create_sample_patients(conn, hospital_id):
    c = conn.cursor()
    # P1 assigned to ICU001
    p1 = ('PAT001', 'Ramesh Kumar', 67, 140, 100, 70, 90, 39.5, 'B+', 'Cardiac', 'high', 'critical', 'icu', 80, 'allocated', 'ICU001', datetime.now().strftime('%Y-%m-%d'), None, 7, hospital_id, 0, 'High Risk Cluster')
    try:
        c.execute('''INSERT OR IGNORE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', p1)
        c.execute("UPDATE beds SET status='occupied', patient_id='PAT001' WHERE id='ICU001'")
    except: pass

# --- ROUTES ---

@app.route('/static/<path:filename>')
def static_files_route(filename): return send_from_directory('static', filename)

@app.route('/')
def home(): return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM hospitals WHERE id=? AND password=?", (data.get('hospital_id'), data.get('password')))
    res = c.fetchone()
    conn.close()
    if res:
        session['hospital_id'] = res[0]
        return jsonify({'success': True, 'hospital_name': res[1]})
    return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/dashboard-data')
def dashboard_data():
    if 'hospital_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    hid = session['hospital_id']
    
    # Stats
    c.execute("SELECT COUNT(*) FROM beds WHERE hospital_id=?", (hid,))
    total_beds = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM beds WHERE hospital_id=? AND status='available'", (hid,))
    avail_beds = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM beds WHERE hospital_id=? AND type='icu'", (hid,))
    icu_total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM patients WHERE hospital_id=? AND severity='high' AND status='allocated'", (hid,))
    critical_load = c.fetchone()[0]
    
    conn.close()
    return jsonify({'stats': {'total_beds': total_beds, 'available_beds': avail_beds, 'icu_beds': icu_total, 'critical_load': critical_load}, 'recent_patients': []})

@app.route('/api/update-capacity', methods=['POST'])
def update_capacity():
    if 'hospital_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    hid = session['hospital_id']
    
    try:
        new_total = int(data['total_beds'])
        new_icu = int(data['icu_beds'])
        new_flex = int(new_total * 0.2)
        new_general = new_total - new_icu - new_flex
        
        if new_general < 0: return jsonify({'success': False, 'message': 'Invalid counts!'})

        conn = sqlite3.connect(DB_NAME)
        conn.execute("UPDATE hospitals SET total_beds=?, icu_beds=? WHERE id=?", (new_total, new_icu, hid))
        
        adjust_bed_capacity(conn, hid, 'general', new_general, 'BED', 'General Ward')
        adjust_bed_capacity(conn, hid, 'icu', new_icu, 'ICU', 'ICU')
        adjust_bed_capacity(conn, hid, 'flexible', new_flex, 'FLEX', 'Flex Ward')
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Hospital capacity updated successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally: conn.close()

@app.route('/api/patient-details/<patient_id>')
def patient_details(patient_id):
    if 'hospital_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('''SELECT name, age, heart_rate, bp_systolic, bp_diastolic, spO2, temperature, blood_group, condition, severity, health_risk, doctor_recommendation, priority_score, status, bed_id, admission_date, expected_stay_days, extended_stay, risk_flag, id FROM patients WHERE id=?''', (patient_id,))
    p = c.fetchone()
    conn.close()
    if not p: return jsonify({'success': False}), 404
    return jsonify({'success': True, 'details': {
        'name': p[0], 'age': p[1], 'heart_rate': p[2], 'bp_systolic': p[3], 'bp_diastolic': p[4],
        'spO2': p[5], 'temperature': p[6], 'blood_group': p[7], 'condition': p[8],
        'ml_severity': p[9], 'triage_risk_level': p[10], 'doctor_recommendation': p[11],
        'priority_score': p[12], 'status': p[13], 'bed_id': p[14], 'admission_date': p[15],
        'expected_stay_days': p[16], 'extended_stay_count': p[17], 'risk_flag': p[18], 'patient_id': p[19]
    }})

@app.route('/api/allocate-bed', methods=['POST'])
def allocate_bed():
    if 'hospital_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    hid = session['hospital_id']
    
    ml_data = {
        'age': data.get('age', 0), 'heart_rate': data.get('heart_rate', 0),
        'blood_pressure_systolic': data.get('blood_pressure_systolic', 0),
        'blood_pressure_diastolic': data.get('blood_pressure_diastolic', 0),
        'spO2': data.get('spO2', 0), 'temperature': data.get('temperature', 0),
        'condition': data.get('admission_cause', '')
    }
    
    sev, msg, _ = predict_severity_ml(ml_data)
    bonus, flag = run_unsupervised_model(ml_data)
    score = calculate_priority_score(sev, data['health_risk'], data['doctor_recommendation'], bonus)
    
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    bed_id, explain = solve_bed_csp(c, hid, score, sev, data['doctor_recommendation'])
    
    c.execute("SELECT COUNT(*) FROM patients")
    pid = f"PAT{c.fetchone()[0]+1:03d}"
    stay = 7 if sev == 'high' else (5 if sev == 'medium' else 2)
    status = 'allocated' if bed_id else 'waiting'
    
    c.execute('''INSERT INTO patients (id, name, age, heart_rate, bp_systolic, bp_diastolic, spO2, temperature, blood_group, condition, severity, health_risk, doctor_recommendation, priority_score, status, bed_id, admission_date, expected_stay_days, hospital_id, risk_flag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', 
              (pid, data['patient_name'], ml_data['age'], ml_data['heart_rate'], ml_data['blood_pressure_systolic'], ml_data['blood_pressure_diastolic'], ml_data['spO2'], ml_data['temperature'], data['blood_group'], data['admission_cause'], sev, data['health_risk'], data['doctor_recommendation'], score, status, bed_id, datetime.now().strftime('%Y-%m-%d'), stay, hid, flag))
    
    if bed_id:
        c.execute("UPDATE beds SET status='occupied', patient_id=?, last_occupied_date=? WHERE id=?", (pid, datetime.now().strftime('%Y-%m-%d'), bed_id))
    
    conn.commit(); conn.close()
    return jsonify({
        'success': True if bed_id else False,
        'message': f"{explain} (ML: {sev}, Risk: {flag})",
        'bed_id': bed_id, 'patient_id': pid, 'ml_severity': sev, 'risk_flag': flag
    })

@app.route('/api/allocated-patients')
def allocated_patients():
    if 'hospital_id' not in session: return jsonify({'error': 'Not logged in'}), 401
    conn = sqlite3.connect(DB_NAME); cursor = conn.cursor()
    
    # 🌟 FIX: Join with beds table to fetch REAL bed type
    cursor.execute('''
        SELECT p.id, p.name, p.age, p.blood_group, p.condition, p.bed_id, 
               p.admission_date, p.severity, p.expected_stay_days, p.extended_stay, 
               p.doctor_recommendation, p.risk_flag, p.heart_rate, p.spO2, p.temperature,
               b.type 
        FROM patients p
        LEFT JOIN beds b ON p.bed_id = b.id
        WHERE p.hospital_id = ? AND p.status = 'allocated' 
        ORDER BY p.admission_date DESC
    ''', (session['hospital_id'],))
    
    patients = []
    for p in cursor.fetchall():
        patients.append({
            'id': p[0], 'name': p[1], 'age': p[2], 'blood_group': p[3], 'condition': p[4],
            'bed_id': p[5], 'admission_date': p[6], 'severity': p[7], 'expected_stay': p[8],
            'extended_stay': p[9], 
            'bed_type': p[15] if p[15] else p[10], # Use actual type from JOIN, or fallback to recommendation
            'risk_flag': p[11],
            'heart_rate': p[12], 'spO2': p[13], 'temperature': p[14],
            'expected_discharge': calculate_expected_discharge(p[6], p[8]),
            'can_extend': (p[15] == 'flexible' and p[9] < 2) # Check actual bed type
        })
    conn.close()
    return jsonify({'patients': patients})

@app.route('/api/discharge-patient', methods=['POST'])
def discharge_patient():
    data = request.json
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('SELECT bed_id FROM patients WHERE id=?', (data.get('patient_id'),))
    res = c.fetchone()
    if res and res[0]: c.execute("UPDATE beds SET status='available', patient_id=NULL WHERE id=?", (res[0],))
    c.execute("UPDATE patients SET status='discharged' WHERE id=?", (data.get('patient_id'),))
    conn.commit(); conn.close()
    return jsonify({'success': True})

@app.route('/api/available-beds')
def available_beds():
    if 'hospital_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute("SELECT id, type, ward, status, last_occupied_date FROM beds WHERE hospital_id=? ORDER BY type, id", (session['hospital_id'],))
    beds = [{'id':b[0], 'type':b[1], 'ward':b[2], 'status':b[3], 'last_occupied':b[4]} for b in c.fetchall()]
    conn.close()
    return jsonify({'beds': beds})

@app.route('/api/extend-stay', methods=['POST'])
def extend_stay():
    data = request.json
    conn = sqlite3.connect(DB_NAME); c = conn.cursor()
    c.execute('SELECT expected_stay_days, extended_stay, doctor_recommendation FROM patients WHERE id=?', (data.get('patient_id'),))
    p = c.fetchone()
    if p and p[2] == 'flexible' and p[1] < 2:
        c.execute('UPDATE patients SET expected_stay_days=?, extended_stay=? WHERE id=?', (p[0]+2, p[1]+1, data.get('patient_id')))
        conn.commit(); conn.close()
        return jsonify({'success': True, 'new_stay_days': p[0]+2})
    conn.close()
    return jsonify({'success': False})

@app.route('/logout')
def logout(): session.clear(); return jsonify({'success': True})

if __name__ == '__main__':
    init_db()
    print(f"🚀 SERBAS AI Running (DB: {DB_NAME})")
    port = int(os.environ.get("PORT", 8001))
    app.run(host='0.0.0.0', port=port)