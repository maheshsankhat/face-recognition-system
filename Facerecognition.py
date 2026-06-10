"""
Face Recognition Attendance System  (Optimized)
================================================
Uses InsightFace (ArcFace) for high-accuracy recognition.
Optimized:  lazy imports, single resource_path, no duplicate code,
            efficient embedding mean-vector per person for fast matching.

Requirements:
    pip install insightface onnxruntime opencv-python pillow numpy
"""

import os
import sys
import threading
import time
from collections import defaultdict, deque

# ── resource_path (PyInstaller support) ──────────────────────────────────────
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
DATASET_PATH      = "dataset"
EMBEDDINGS_FILE   = "embeddings.npy"
COSINE_THRESHOLD  = 0.40
IMAGES_PER_PERSON = 50
SAVE_EVERY_MS     = 200

# ══════════════════════════════════════════════════════════════
#  InsightFace — loaded once at startup
# ══════════════════════════════════════════════════════════════
import tkinter as tk
from tkinter import messagebox

print("Loading ArcFace model…")
try:
    import insightface
    from insightface.app import FaceAnalysis
    import numpy as np
    import cv2
    from PIL import Image, ImageTk

    model_root = resource_path(".insightface")
    models_path = os.path.join(model_root, "models", "buffalo_l")
    
    _fa = FaceAnalysis(
        name="buffalo_l",
        root=model_root,           # best accuracy / speed trade-off
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )
    _fa.prepare(ctx_id=-1, det_size=(224, 224))
    print("ArcFace model loaded ✅")

except Exception as e:
    import traceback
    traceback.print_exc()

    messagebox.showerror(
        "Import Error",
        str(e)
    )
    sys.exit(1)
# ══════════════════════════════════════════════════════════════
#  EMBEDDING HELPERS
# ══════════════════════════════════════════════════════════════

def cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def load_embeddings() -> dict:
    if os.path.exists(EMBEDDINGS_FILE):
        return np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
    return {}


def save_embeddings(embeddings: dict):
    np.save(EMBEDDINGS_FILE, embeddings)


def recognize_face(embedding: "np.ndarray", embeddings: dict):
    """Returns (name, score). Uses mean embedding per person → O(persons) not O(all images)."""
    best_name, best_score = "Unknown", 0.0
    for name, stored_list in embeddings.items():
        # compare against mean vector for speed
        mean_emb = np.mean(stored_list, axis=0).astype(np.float32)
        score = cosine_sim(embedding, mean_emb)
        if score > best_score:
            best_score, best_name = score, name
    if best_score < COSINE_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score

# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("1200x700")
root.configure(bg="#1e1e1e")

tk.Label(
    root, text="Face Recognition Attendance System",
    font=("Arial", 24, "bold"), bg="#1e1e1e", fg="white",
).pack(pady=10)

sidebar = tk.Frame(root, bg="#111111", width=200)
sidebar.pack(side="left", fill="y")

main = tk.Frame(root, bg="#1e1e1e")
main.pack(side="right", expand=True, fill="both")

cap_global = [None]


def clear_main():
    for w in main.winfo_children():
        w.destroy()


def release_cap():
    if cap_global[0]:
        cap_global[0].release()
        cap_global[0] = None


def open_camera(width=640, height=480, fps=30):
    release_cap()
    c = cv2.VideoCapture(0)
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c.set(cv2.CAP_PROP_FPS,          fps)
    c.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap_global[0] = c
    return c

# ══════════════════════════════════════════════════════════════
#  CAPTURE PAGE
# ══════════════════════════════════════════════════════════════
def upload_page():
    release_cap()
    clear_main()
    count = [0]

    tk.Label(main, text="Capture Face Dataset",
             font=("Arial", 22, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)
    tk.Label(main, text="Enter Person Name",
             font=("Arial", 14), bg="#1e1e1e", fg="white").pack()

    name_entry = tk.Entry(main, font=("Arial", 14), width=25)
    name_entry.pack(pady=10)

    count_lbl = tk.Label(main, text=f"Images: 0 / {IMAGES_PER_PERSON}",
                         font=("Arial", 12), bg="#1e1e1e", fg="lightgreen")
    count_lbl.pack()

    camera_frame = tk.Frame(main, bg="black", width=700, height=500)
    camera_frame.pack(pady=10)
    video_label = tk.Label(camera_frame)
    video_label.pack()

    last_save = [0]

    def start_capture():
        person_name = name_entry.get().strip()
        if not person_name:
            messagebox.showwarning("Warning", "Please enter a person name.")
            return

        person_path = os.path.join(DATASET_PATH, person_name)
        os.makedirs(person_path, exist_ok=True)
        for f in os.listdir(person_path):
            if f.lower().endswith(".jpg"):
                os.remove(os.path.join(person_path, f))
        count[0] = 0

        open_camera(640, 480)

        def update():
            if cap_global[0] is None:
                return
            ret, frame = cap_global[0].read()
            if not ret:
                video_label.after(20, update)
                return

            frame = cv2.flip(frame, 1)
            total = IMAGES_PER_PERSON
            pct = count[0] / total
            instruction = (
                "Look Straight"       if pct < 0.4 else
                "Turn Slightly Left"  if pct < 0.6 else
                "Turn Slightly Right" if pct < 0.8 else
                "Natural Expression"
            )
            cv2.putText(frame, instruction, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            now_ms = int(time.time() * 1000)
            if count[0] < IMAGES_PER_PERSON and (now_ms - last_save[0]) >= SAVE_EVERY_MS:
                faces = _fa.get(frame)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    x1,y1,x2,y2 = [int(v) for v in face.bbox]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    count[0] += 1
                    cv2.imwrite(os.path.join(person_path, f"{count[0]}.jpg"), frame)
                    last_save[0] = now_ms
                    count_lbl.config(text=f"Images: {count[0]} / {IMAGES_PER_PERSON}")

            cv2.putText(frame, f"Images: {count[0]}/{IMAGES_PER_PERSON}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize((700, 500)))
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            if count[0] >= IMAGES_PER_PERSON:
                release_cap()
                messagebox.showinfo("Done", "Dataset capture complete! ✅\nNow click 'Enroll Person'.")
                return

            video_label.after(20, update)

        update()

    tk.Button(main, text="Start Face Capture",
              font=("Arial", 14, "bold"), bg="#333333", fg="white",
              padx=20, pady=10, command=start_capture).pack(pady=15)

# ══════════════════════════════════════════════════════════════
#  ENROLL PAGE
# ══════════════════════════════════════════════════════════════
def enroll_page():
    clear_main()
    tk.Label(main, text="Enroll Persons",
             font=("Arial", 22, "bold"), bg="#1e1e1e", fg="white").pack(pady=30)

    status_lbl = tk.Label(main, text="", font=("Arial", 14), bg="#1e1e1e", fg="lightgreen")
    status_lbl.pack(pady=20)
    detail_lbl = tk.Label(main, text="", font=("Arial", 12), bg="#1e1e1e", fg="white")
    detail_lbl.pack()

    def start_enroll():
        enroll_btn.config(state="disabled")
        status_lbl.config(text="Enrolling…", fg="yellow")

        def do_enroll():
            if not os.path.isdir(DATASET_PATH):
                root.after(0, lambda: status_lbl.config(text="Dataset folder not found!", fg="red"))
                root.after(0, lambda: enroll_btn.config(state="normal"))
                return

            people = sorted(p for p in os.listdir(DATASET_PATH)
                            if os.path.isdir(os.path.join(DATASET_PATH, p)))
            if not people:
                root.after(0, lambda: status_lbl.config(text="No persons found!", fg="red"))
                root.after(0, lambda: enroll_btn.config(state="normal"))
                return

            embeddings = {}
            total_embs = 0

            for person_name in people:
                root.after(0, lambda n=person_name: detail_lbl.config(text=f"Processing: {n}"))
                path = os.path.join(DATASET_PATH, person_name)
                person_embs = []

                for fn in sorted(os.listdir(path)):
                    if not fn.lower().endswith(".jpg"):
                        continue
                    img = cv2.imread(os.path.join(path, fn))
                    if img is None:
                        continue
                    try:
                        faces = _fa.get(img)
                    except Exception:
                        continue
                    if not faces:
                        continue
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    person_embs.append(face.embedding.astype(np.float32))

                if person_embs:
                    embeddings[person_name] = person_embs
                    total_embs += len(person_embs)

            if not embeddings:
                root.after(0, lambda: status_lbl.config(text="No faces detected!", fg="red"))
                root.after(0, lambda: enroll_btn.config(state="normal"))
                return

            save_embeddings(embeddings)
            root.after(0, lambda: status_lbl.config(
                text=f"✅ Enrolled {len(embeddings)} person(s)  |  {total_embs} embeddings",
                fg="lightgreen"))
            root.after(0, lambda: detail_lbl.config(text=""))
            root.after(0, lambda: enroll_btn.config(state="normal"))

        threading.Thread(target=do_enroll, daemon=True).start()

    enroll_btn = tk.Button(main, text="Enroll All Persons",
                           font=("Arial", 14, "bold"), bg="#333333", fg="white",
                           padx=20, pady=10, command=start_enroll)
    enroll_btn.pack(pady=20)

    embeddings = load_embeddings()
    if embeddings:
        tk.Label(main, text="Currently enrolled:",
                 font=("Arial", 12, "bold"), bg="#1e1e1e", fg="white").pack(pady=(20,5))
        for name, embs in embeddings.items():
            tk.Label(main, text=f"  • {name}  ({len(embs)} embeddings)",
                     font=("Arial", 12), bg="#1e1e1e", fg="lightgreen").pack()

# ══════════════════════════════════════════════════════════════
#  RECOGNITION PAGE
# ══════════════════════════════════════════════════════════════
def recognition_page():
    clear_main()
    tk.Label(main, text="Real-Time Face Recognition",
             font=("Arial", 22, "bold"), bg="#1e1e1e", fg="white").pack(pady=20)

    status_lbl = tk.Label(main, text="", font=("Arial", 12), bg="#1e1e1e", fg="yellow")
    status_lbl.pack(pady=5)

    if not os.path.exists(EMBEDDINGS_FILE):
        tk.Label(main, text="No enrolled faces found!\nCapture dataset and enroll first.",
                 font=("Arial", 14), bg="#1e1e1e", fg="red").pack(pady=20)
        return

    embeddings = load_embeddings()
    if not embeddings:
        tk.Label(main, text="Embeddings file is empty. Please re-enroll.",
                 font=("Arial", 14), bg="#1e1e1e", fg="red").pack(pady=20)
        return

    tk.Label(main, text=f"Enrolled: {', '.join(embeddings.keys())}",
             font=("Arial", 11), bg="#1e1e1e", fg="#aaaaaa").pack()

    camera_frame = tk.Frame(main, bg="black", width=640, height=480)
    camera_frame.pack(pady=10)
    video_label = tk.Label(camera_frame)
    video_label.pack()

    cap     = [None]
    running = [False]
    skip    = [0]

    votes     = defaultdict(lambda: deque(maxlen=6))
    last_name = {}
    last_pos  = {}
    cached    = []

    def find_vote_key(cx, cy):
        best_key, best_dist = None, 9999
        for k, (kx, ky) in last_pos.items():
            d = ((cx-kx)**2 + (cy-ky)**2)**0.5
            if d < best_dist:
                best_dist, best_key = d, k
        if best_key and best_dist < 160:
            return best_key
        return (int(cx/20)*20, int(cy/20)*20)

    def start_camera():
        if running[0]:
            return
        open_camera()
        cap[0] = cap_global[0]
        running[0] = True
        status_lbl.config(text="Camera running — ArcFace recognition active")
        loop()

    def stop_camera():
        running[0] = False
        release_cap()
        cap[0] = None
        video_label.configure(image="")
        status_lbl.config(text="Camera stopped.")

    def loop():
        if not running[0] or cap[0] is None:
            return

        ret, frame = cap[0].read()
        if not ret:
            video_label.after(20, loop)
            return

        frame    = cv2.flip(frame, 1)
        skip[0] += 1

        if skip[0] % 10 == 0:
            try:
                faces_detected = _fa.get(frame)
            except Exception:
                faces_detected = []

            cached.clear()
            current_keys = set()

            for face in faces_detected:
                x1,y1,x2,y2 = [int(v) for v in face.bbox]
                w, h = x2-x1, y2-y1
                if w < 60 or h < 60:
                    continue

                emb = face.embedding.astype(np.float32)
                name, score = recognize_face(emb, embeddings)
                cx, cy = x1 + w//2, y1 + h//2
                key = find_vote_key(cx, cy)
                current_keys.add(key)

                old_cx, old_cy = last_pos.get(key, (cx, cy))
                cx = int(0.7*old_cx + 0.3*cx)
                cy = int(0.7*old_cy + 0.3*cy)
                last_pos[key] = (cx, cy)

                votes[key].append((name, score))
                name_counts = defaultdict(list)
                for vn, vs in votes[key]:
                    name_counts[vn].append(vs)

                best_voted = max(name_counts, key=lambda n: (len(name_counts[n]), max(name_counts[n])))
                best_score_voted = float(max(name_counts[best_voted]))

                if best_voted != "Unknown" and best_score_voted >= COSINE_THRESHOLD:
                    color = (0,255,0)
                    text  = f"{best_voted}  ({int(best_score_voted*100)}%)"
                    last_name[key] = best_voted
                elif key in last_name:
                    color = (0,200,100)
                    text  = f"{last_name[key]} ?"
                else:
                    color = (0,0,255)
                    text  = "Unknown"

                dup = any(
                    max(0,(min(x2,px2)-max(x1,px1)))*max(0,(min(y2,py2)-max(y1,py1))) / max(w*h,1) > 0.50
                    for px1,py1,px2,py2,_,_ in cached
                )
                if not dup:
                    cached.append((x1,y1,x2,y2,text,color))

            for k in list(votes):
                if k not in current_keys:
                    del votes[k]; last_name.pop(k,None); last_pos.pop(k,None)

        for (x1,y1,x2,y2,text,color) in cached:
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1,y1-th-12), (x1+tw+6,y1), color, -1)
            cv2.putText(frame, text, (x1+3,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(20, loop)

    btn_frame = tk.Frame(main, bg="#1e1e1e")
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Start Recognition",
              font=("Arial",12,"bold"), bg="#333333", fg="white",
              padx=20, pady=10, command=start_camera).grid(row=0, column=0, padx=10)
    tk.Button(btn_frame, text="Stop Camera",
              font=("Arial",12,"bold"), bg="red", fg="white",
              padx=20, pady=10, command=stop_camera).grid(row=0, column=1, padx=10)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
for txt, cmd in [
    ("📁 Capture Dataset",  upload_page),
    ("🧠 Enroll Persons",   enroll_page),
    ("📊 Face Recognition", recognition_page),
]:
    tk.Button(sidebar, text=txt, font=("Arial",12),
              bg="#333333", fg="white", width=20, pady=10,
              command=cmd).pack(pady=10)

upload_page()
root.mainloop()