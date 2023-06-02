from flask import Flask, render_template, request, jsonify
import cv2
import os
import face_recognition
from TDDFA import TDDFA

app = Flask(__name__)

# Step 1: Application Initialization
@app.route('/')
def index():
    return render_template('index.html')

# Step 2: Student Registration
@app.route('/student_registration', methods=['GET', 'POST'])
def student_registration():
    if request.method == 'POST':
        # Get student details from the registration form
        name = request.form['name']
        roll_number = request.form['roll_number']
        academic_program = request.form['academic_program']
        enrolled_courses = request.form['enrolled_courses']

        # Create a directory for the student using roll number as the folder name
        student_directory = f'students/{roll_number}'
        os.makedirs(student_directory, exist_ok=True)

        # Save the student details to a text file in the student directory
        with open(f'{student_directory}/details.txt', 'w') as f:
            f.write(f'Name: {name}\n')
            f.write(f'Roll Number: {roll_number}\n')
            f.write(f'Academic Program: {academic_program}\n')
            f.write(f'Enrolled Courses: {enrolled_courses}\n')

        return 'Registration Successful'
    else:
        return render_template('student_registration.html')

# Step 3: Face Extraction and Detection
@app.route('/face_extraction', methods=['GET', 'POST'])
def face_extraction():
    if request.method == 'POST':
        # Get the video file uploaded during registration
        video = request.files['video']

        # Create a directory for face extraction
        extraction_directory = 'face_extraction'
        os.makedirs(extraction_directory, exist_ok=True)

        # Save the uploaded video to the extraction directory
        video_path = f'{extraction_directory}/video.mp4'
        video.save(video_path)

        # Extract frames from the video
        frames_directory = f'{extraction_directory}/frames'
        os.makedirs(frames_directory, exist_ok=True)
        extract_frames(video_path, frames_directory)

        # Perform face detection on the extracted frames
        detected_faces_directory = f'{extraction_directory}/detected_faces'
        os.makedirs(detected_faces_directory, exist_ok=True)
        detect_faces(frames_directory, detected_faces_directory)

        # Generate multiple faces from different angles using TDDFA
        generated_faces_directory = f'{extraction_directory}/generated_faces'
        os.makedirs(generated_faces_directory, exist_ok=True)
        generate_faces(detected_faces_directory, generated_faces_directory)

        return 'Face Extraction and Generation Complete'
    else:
        return render_template('face_extraction.html')

# Step 4: Student Login
@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        # Get student login credentials from the login form
        roll_number = request.form['roll_number']

        # Check if the student directory exists
        student_directory = f'students/{roll_number}'
        if os.path.isdir(student_directory):
            return 'Login Successful'
        else:
            return 'Invalid Credentials'
    else:
        return render_template('student_login.html')

# Step 5: Teacher Login
@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        # Get teacher login credentials from the login form
        username = request.form['username']
        password = request.form['password']

        # Check if the credentials are valid
        if username == 'admin' and password == 'admin123':
            return 'Login Successful'
        else:
            return 'Invalid Credentials'
    else:
        return render_template('teacher_login.html')

# Step 6: Student Dashboard
@app.route('/student_dashboard')
def student_dashboard():
    return render_template('student_dashboard.html')

# Step 7: Teacher Dashboard
@app.route('/teacher_dashboard')
def teacher_dashboard():
    return render_template('teacher_dashboard.html')

# Step 8: Take Attendance
@app.route('/take_attendance')
def take_attendance():
    return render_template('take_attendance.html')

# Step 9: Manual Attendance
@app.route('/manual_attendance')
def manual_attendance():
    return render_template('manual_attendance.html')

# Step 10: Attendance Report
@app.route('/attendance_report')
def attendance_report():
    return render_template('attendance_report.html')

# Step 11: Logout
@app.route('/logout')
def logout():
    return render_template('logout.html')

# Helper function to extract frames from a video
def extract_frames(video_path, output_directory):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read and save frames until the video is finished
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_path = os.path.join(output_directory, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release the video file
    video.release()

# Helper function to detect faces in the frames
def detect_faces(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)
            face_locations = face_recognition.face_locations(image)

            for i, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                face_image = image[top:bottom, left:right]

                face_filename = f'{os.path.splitext(filename)[0]}_{i}.jpg'
                output_path = os.path.join(output_directory, face_filename)
                cv2.imwrite(output_path, face_image)

# Helper function to generate faces from different angles using TDDFA
    tddfa = TDDFA()
  
def generate_faces(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)

            # Perform 3D face reconstruction using TDDFA
            vertices, landmarks = tddfa(image)

            # Generate new views by changing the camera positions
            num_views = 4  # Number of desired views
            for i in range(num_views):
                # Rotate the camera around the face
                rotation_matrix = get_rotation_matrix(i * 360 / num_views)

                # Render the 3D face model with the new camera position
                rendered_image = render_face_model(vertices, landmarks, rotation_matrix)

                # Save the rendered image to the output directory
                output_path = os.path.join(output_directory, f'{i}_{filename}')
                cv2.imwrite(output_path, rendered_image)

def get_rotation_matrix(angle):
    # Convert the angle to radians
    radian_angle = np.radians(angle)

    # Define the rotation axis (e.g., [0, 1, 0] for rotation around the y-axis)
    rotation_axis = np.array([0, 1, 0])

    # Compute the rotation matrix using the Rodrigues formula
    rotation_matrix, _ = cv2.Rodrigues(radian_angle * rotation_axis)

    return rotation_matrix

def render_face_model(vertices, landmarks, rotation_matrix):
    # Apply the rotation matrix to the vertices
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    # Render the face model with the rotated vertices
    # You can use libraries like OpenGL or Pyrender for rendering

    return rendered_image

            # Save the generated faces to the output directory
  generated_faces_path = os.path.join(output_directory, filename)
            cv2.imwrite(generated_faces_path, image)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
