from flask import Flask, render_template, Response, redirect, url_for, request, send_file
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import time
from datetime import datetime
from pymongo import MongoClient
import bson
import tempfile
from bson.objectid import ObjectId
from collections import deque
import mediapipe as mp
import pyrealsense2 as rs
import os
import signal
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)
app.secret_key = 'secret'

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["physiodb"]
physiotable = db["physiotable"]

# RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align = rs.align(rs.stream.color)

# MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Aruco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
PERIMETER_CM = 46

# App state
streaming = {"active": False}
selected_side = {"value": "both"}
duration = {"value": None}
start_time = {"value": None}
frame_numbers = deque(maxlen=300)
jump_heights = {"left": deque(maxlen=300), "right": deque(maxlen=300)}
initial_foot_pixels = {"left": None, "right": None}
frame_count = 0
video_filename = "session_video.mp4"


def get_median_depth(depth_frame, pixel, kernel_size=5):
    x, y = pixel
    width, height = depth_frame.get_width(), depth_frame.get_height()
    depth_values = []
    for dx in range(-kernel_size // 2, kernel_size // 2 + 1):
        for dy in range(-kernel_size // 2, kernel_size // 2 + 1):
            px = x + dx
            py = y + dy
            if 0 <= px < width and 0 <= py < height:
                d = depth_frame.get_distance(px, py)
                if d > 0:
                    depth_values.append(d)
    return np.median(depth_values) * 100 if depth_values else 0


def generate_graph():
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    frame_list = list(frame_numbers)

    for i, side in enumerate(["right", "left"]):
        y = list(jump_heights[side])
        if y:
            x = frame_list[-len(y):] if len(frame_list) >= len(y) else list(range(len(y)))
            axs[i].plot(x, y, label=side.capitalize(), color=("green" if side == "right" else "orange"))
            axs[i].set_title(f"{side.capitalize()} Leg")
            axs[i].grid(True)

    axs[0].set_ylabel("Height (cm)")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Height (cm)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


#@app.route('/')
#def index():
#    return render_template('index.html', streaming=streaming["active"], side=selected_side["value"])


@app.route('/start/<which>')
def start(which):
    selected_side["value"] = which
    streaming["active"] = True
    seconds = request.args.get("seconds")
    duration["value"] = float(seconds) if seconds else None
    return redirect(url_for('track'))


@app.route('/stop')
def stop():
    streaming["active"] = False
    duration["value"] = None
    selected_side["value"] = None
    return redirect(url_for('track'))


@app.route('/save', methods=['POST'])
def save():
    patient_id = request.form.get("patient_id", "unknown")
    graph_data = generate_graph()
    if not os.path.exists(video_filename):
        return "No video recorded", 400
    with open(video_filename, "rb") as f:
        video_data = f.read()

    doc = {
        "patient_id": patient_id,
        "timestamp": datetime.utcnow(),
        "duration": duration["value"],
        "side": selected_side["value"],
        "frame_numbers": list(frame_numbers),
        "heights": {k: list(v) for k, v in jump_heights.items()},
        "graph_image": bson.Binary(graph_data),
        "video_file": bson.Binary(video_data)
    }
    physiotable.insert_one(doc)
    return redirect(url_for('track'))


@app.route('/sessions')
def sessions_page():
    patient_id = request.args.get("patient_id")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    if not patient_id and not start_date and not end_date:
        return render_template("sessions.html", sessions=[])

    query = {}
    if patient_id:
        query["patient_id"] = patient_id
    '''if start_date:
        query["timestamp"] = {"$gte": datetime.strptime(start_date, "%Y-%m-%d")}
    if end_date:
        if "timestamp" not in query:
            query["timestamp"] = {}
        query["timestamp"]["$lte"] = datetime.strptime(end_date, "%Y-%m-%d")
    '''
    if start_date:
        if "timestamp" not in query:
            query["timestamp"] = {}
        query["timestamp"] = {"$gte": datetime.strptime(start_date, "%Y-%m-%dT%H:%M")}
    if end_date:
        if "timestamp" not in query:
            query["timestamp"] = {}
        query["timestamp"]["$lte"] = datetime.strptime(end_date, "%Y-%m-%dT%H:%M")

    results = list(physiotable.find(query).sort("timestamp", -1))
    return render_template("sessions.html", sessions=results)

@app.route('/graph/<session_id>')
def get_graph(session_id):
    session = physiotable.find_one({"_id": ObjectId(session_id)})
    if session and "graph_image" in session:
        return send_file(io.BytesIO(session["graph_image"]),
                         mimetype='image/jpeg',
                         as_attachment=False,
                         download_name="graph.jpg")
    return "Graph not found", 404


@app.route('/video/<session_id>')
def get_video(session_id):
    session = physiotable.find_one({"_id": ObjectId(session_id)})
    if session and "video_file" in session:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(session["video_file"])
        tmp.close()
        return send_file(tmp.name,
                         mimetype='video/mp4',
                         as_attachment=True,
                         download_name="session_video.mp4")
    return "Video not found", 404


@app.route('/video_feed')
def video_feed():
    def generate():
        global frame_count
        try:
            pipeline.stop()
        except Exception:
            pass

        try:
            pipeline.start(config)
        except Exception as e:
            yield b""
            return

        video_writer = cv2.VideoWriter(video_filename,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       30,
                                       (640, 480))

        align = rs.align(rs.stream.color)
        frame_count = 0
        frame_numbers.clear()
        for side in ["left", "right"]:
            jump_heights[side].clear()
            initial_foot_pixels[side] = None

        start_time["value"] = time.time()

        while streaming["active"]:
            if duration["value"] and time.time() - start_time["value"] > duration["value"]:
                break

            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            color_image = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco_detector.detectMarkers(gray)
            pixel_cm_ratio = None
            tag_depth = None

            if ids is not None:
                pts = corners[0][0]
                aruco_perimeter = cv2.arcLength(pts, True)
                pixel_cm_ratio = aruco_perimeter / PERIMETER_CM
                tag_center = np.mean(pts, axis=0).astype(int)
                tag_depth = get_median_depth(depth, tuple(tag_center))

            results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            sides = []
            if selected_side["value"] in ["left", "both"]:
                sides.append("left")
            if selected_side["value"] in ["right", "both"]:
                sides.append("right")

            if results.pose_landmarks:
                for side in sides:
                    lm = results.pose_landmarks.landmark
                    points = {
                        "left": (mp_pose.PoseLandmark.LEFT_HEEL,
                                 mp_pose.PoseLandmark.LEFT_ANKLE,
                                 mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                        "right": (mp_pose.PoseLandmark.RIGHT_HEEL,
                                  mp_pose.PoseLandmark.RIGHT_ANKLE,
                                  mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
                    }
                    heel, ankle, foot = points[side]
                    x = int(((lm[heel].x + lm[ankle].x + lm[foot].x) / 3) * 640)
                    y = int(((lm[heel].y + lm[ankle].y + lm[foot].y) / 3) * 480)

                    if initial_foot_pixels[side] is None:
                        initial_foot_pixels[side] = (x, y)

                    if pixel_cm_ratio and tag_depth:
                        current_depth = get_median_depth(depth, (x, y))
                        base_depth = get_median_depth(depth, initial_foot_pixels[side])
                        if current_depth and base_depth:
                            corrected_ratio = pixel_cm_ratio * (tag_depth / ((current_depth + base_depth) / 2))
                            pixels = abs(initial_foot_pixels[side][1] - y)
                            height_cm = pixels / corrected_ratio
                            jump_heights[side].append(height_cm)
                        else:
                            jump_heights[side].append(0)
                    else:
                        jump_heights[side].append(0)

            mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_count += 1
            frame_numbers.append(frame_count)

            cv2.putText(color_image,
                        f"Tracking: {selected_side['value']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

            video_writer.write(color_image)
            ret, buffer = cv2.imencode(".jpg", color_image)
            if not ret:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        video_writer.release()
        pipeline.stop()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/track')
def track():
    return render_template('track.html', streaming=streaming["active"], side=selected_side["value"])

@app.route('/graph_feed')
def graph_feed():
    def stream():
        while streaming["active"]:
            img = generate_graph()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + img + b"\r\n")
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return "Shutting down..."

@app.route('/about')
def about():
    return render_template('about.html')

from bson.objectid import ObjectId

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    sessions = list(physiotable.find().sort("timestamp", -1))  # Show all sessions
    selected_sessions = []
    comparison_insights = None

    if request.method == 'POST':
        session_id1 = request.form.get('session_id1')
        session_id2 = request.form.get('session_id2')

        # Fetch sessions by ObjectId
        session1 = physiotable.find_one({"_id": ObjectId(session_id1)})
        session2 = physiotable.find_one({"_id": ObjectId(session_id2)})

        selected_sessions = [s for s in [session1, session2] if s]

        if len(selected_sessions) == 2:
            genai.configure(api_key="AIzaSyAxjfq5HIvWxfgnlKcPXVHAt8tSb93GUJo")
            model = genai.GenerativeModel("gemini-1.5-flash-latest")

            images = []
            for s in selected_sessions:
                img_bytes = s['graph_image']
                pil_image = Image.open(io.BytesIO(img_bytes))
                images.append(pil_image)

            prompt = "Compare these two jump height graphs and provide insights."
            response = model.generate_content([*images, prompt])
            comparison_insights = response.text if hasattr(response, 'text') else "No insights generated."

    return render_template('compare.html', sessions=sessions, selected_sessions=selected_sessions, insights=comparison_insights)


@app.route('/insights', methods=['GET', 'POST'])
def insights():
    session = None
    insights_text = None

    if request.method == 'POST':
        session_id = request.form.get('session_id')
        if not session_id:
            return "No session ID provided", 400

        session = physiotable.find_one({"_id": ObjectId(session_id)})
        if not session or "graph_image" not in session:
            return "Session or graph not found", 404

        genai.configure(api_key="AIzaSyAxjfq5HIvWxfgnlKcPXVHAt8tSb93GUJo")
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        pil_image = Image.open(io.BytesIO(session['graph_image']))
        response = model.generate_content([pil_image, "Provide insights about this jump height graph."])
        insights_text = response.text if hasattr(response, 'text') else "No insights generated."

    return render_template('insights.html', session=session, insights=insights_text)


if __name__ == '__main__':
    app.run(debug=True)

