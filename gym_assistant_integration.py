import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from GymAssistantClassifier import GymAssistantClassifier
import numpy as np
from collections import deque
import math


# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
# https://github.com/KDMStromback/mm-fit/blob/master/EDA.ipynb
# https://google.github.io/mediapipe/solutions/pose.html
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}

count = {'squats': 0.0,
 'lunges': 0.0,
 'bicep_curls': 0.0,
 'situps': 0.0,
 'pushups': 0.0,
 'tricep_extensions': 0.0,
 'dumbbell_rows': 0.0,
 'jumping_jacks': 0.0,
 'dumbbell_shoulder_press': 0.0,
 'lateral_shoulder_raises': 0.0}

#https://google.github.io/mediapipe/solutions/pose.html``
def set_pose_parameters():
    mode = False  # For video feed input and localize the pose landmark once it is detected on video
    complexity = 1 # Landmark accuracy [0,1,2] as well as inferece latency
    smooth_landmarks = True # filters pose landmarks across different input images to reduce jitter
    enable_segmentation = False # if true Solution also generates the segmentation mask
    smooth_segmentation = True # filters segmentation masks across different input images to reduce jitte
    detectionCon = 0.5 # detection confidence
    trackCon = 0.5 # tracking confidence 
    mpPose = mp.solutions.pose 
    return mode,complexity,smooth_landmarks,enable_segmentation,smooth_segmentation,detectionCon,trackCon,mpPose

# Provides img from cv2, RGB converted image, set default to draw landmart true function to perform : #Get pose and draw landmarks.
# https://google.github.io/mediapipe/solutions/pose.html#pose_landmarks
def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

# Get Position of landmark in list from mediapipe : 33 landmarks 
def get_position(img, results, height, width, draw=True ):
        landmark_list = []
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                #finding height, width of the image printed
                height, width, c = img.shape
                #Determining the pixels of the landmarks
                landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
                # RGB (255,0,0) : red
                if draw:
                    cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255,0,0), cv2.FILLED)
        return landmark_list    


# To Draw the line joining 3 points, angels, and 2 circle according to points given in function
def get_angle(img, landmark_list, point1, point2, point3, draw=True):   
        #Retrieve landmark coordinates from point identifiers
        x1, y1 = landmark_list[point1][1:]
        x2, y2 = landmark_list[point2][1:]
        x3, y3 = landmark_list[point3][1:]
            
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        #Handling angle edge cases: Obtuse and negative angles
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
            
        if draw:
            #Drawing lines between the three points - (255,255,255) :  white
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            #Drawing circles at intersection points of lines - indigo
            """ img (CvArr) – Image where the circle is drawn
                center (CvPoint) – Center of the circle
                radius (int) – Radius of the circle
                color (CvScalar) – Circle color
                thickness (int) – Thickness of the circle outline if positive, otherwise this indicates that a filled circle is to be drawn
                lineType (int) – Type of the circle boundary, see Line description
                shift (int) – Number of fractional bits in the center coordinates and radius value """
            cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
            cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
            cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
            
            #Show angles between lines
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle
 
    
# Convert mediapipe's keypoint to corosponding classifier keypoint for model predication.    
def convert_mediapipe_keypoints_for_model(lm_dict, landmark_list):
    inp_workout = []
    for index in range(0, 36):
        if index < 18:
            inp_workout.append(round(landmark_list[lm_dict[index]][1],3))
        else:
            inp_workout.append(round(landmark_list[lm_dict[index-18]][2],3))
    return inp_workout

# Setting variables for video feed
def set_video_feed_variables():
    cap = cv2.VideoCapture(0) # This will return video from the first webcam on your computer.
    count = {'squats': 0.0,
 'lunges': 0.0,
 'bicep_curls': 0.0,
 'situps': 0.0,
 'pushups': 0.0,
 'tricep_extensions': 0.0,
 'dumbbell_rows': 0.0,
 'jumping_jacks': 0.0,
 'dumbbell_shoulder_press': 0.0,
 'lateral_shoulder_raises': 0.0}
    direction = 0
    form = 0 
    feedback = "Bad Form."
    frame_queue = deque(maxlen=250) # Queue works on FIFO concept with max capacity of 250
    clf = GymAssistantClassifier('exercise_classifier_v3.tflite')
    return cap,count,direction,form,feedback,frame_queue,clf

# Giving corosponding linear interpolation value for pushup and seatups exercise. 
def set_percentage_bar_and_text(elbow_angle, knee_angle,hip_angle,shoulder_angle, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":    
        success_percentage = np.interp(elbow_angle, (90, 160), (0, 100))
        progress_bar = np.interp(elbow_angle, (90, 160), (380, 30))
        return success_percentage,progress_bar
    # Else only handles squats right now
    elif workout_name_after_smoothening == "bicep_curls":
        success_percentage = np.interp(elbow_angle, (60, 160), (0, 100))
        progress_bar = np.interp(elbow_angle, (60, 160), (380, 30))
        return success_percentage,progress_bar

    elif workout_name_after_smoothening == "jumping_jacks":
        success_percentage = np.interp(shoulder_angle, (40, 155), (0, 100))
        progress_bar = np.interp(shoulder_angle, (40, 155), (380, 30))
        return success_percentage,progress_bar

    elif workout_name_after_smoothening.strip() == "situps":
        success_percentage = np.interp(hip_angle, (45, 90), (0, 100))
        progress_bar = np.interp(hip_angle, (45,90), (380, 30))
        return success_percentage,progress_bar
    elif workout_name_after_smoothening == "dumbbell_rows":
        success_percentage = np.interp(elbow_angle, (30, 155), (0, 100))
        progress_bar = np.interp(elbow_angle, (30, 155), (380, 30))
        return success_percentage,progress_bar
    elif workout_name_after_smoothening == "lunges":
        success_percentage = np.interp(knee_angle, (80, 155), (0, 100))
        progress_bar = np.interp(knee_angle, (80, 155), (380, 30))
        return success_percentage,progress_bar
    else:
        success_percentage = np.interp(knee_angle, (90, 160), (0, 100))
        progress_bar = np.interp(knee_angle, (90, 160), (380, 30))
        return success_percentage,progress_bar

# Evaluting important body angle for checking whether person is doing proper exercise or not.
def set_body_angles_from_keypoints(get_angle, img, landmark_list):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
    hip_angle = get_angle(img, landmark_list, 11, 23,25)
    elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16)
    shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24)
    hip_angle_right = get_angle(img, landmark_list, 12, 24,26)
    knee_angle = get_angle(img, landmark_list, 24,26, 28)
    knee_angle_right = get_angle(img, landmark_list, 25,27, 29)
    return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle,knee_angle_right

# Calling function to convert mediapipe perameters to model suitable parameter and predicting workout name.
def set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list):
    inp_workout = convert_mediapipe_keypoints_for_model(lm_dict, landmark_list)
    workout_name = clf.predict(inp_workout)
    frame_queue.append(workout_name)
    workout_name_after_smoothening = max(set(frame_queue), key=frame_queue.count)
    return "Workout Name: " + workout_name_after_smoothening

# following 4 function is for displaying various requirements onto screen.
def draw_percentage_progress_bar(form, img, success_percentage, progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    if form == 1:
        cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
        cv2.rectangle(img, (xd, int(progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

def display_rep_count(count, img):
    xc, yc = 85, 100
    cv2.putText(img, "Reps: " + str(int(count)), (xc, yc), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 2)

def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def show_workout_name_from_model(img, workout_name_after_smoothening):
    xw, yw = 85, 40
    cv2.putText(img, workout_name_after_smoothening, (xw,yw), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

# check wheter the the exerise form is correct ot not  if correct then return 1
def check_form(elbow_angle, shoulder_angle, hip_angle, knee_angle,form, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 :
            form = 1
    # For now, else impleements squats condition        
    elif workout_name_after_smoothening == "squats":
        if knee_angle > 160:
            form = 1
    elif workout_name_after_smoothening == "bicep_curls":
        if elbow_angle > 160:
            form = 1
    elif workout_name_after_smoothening == "jumping_jacks":
        if shoulder_angle > 30 and shoulder_angle < 60 :
            form = 1
            # todo situps
    elif workout_name_after_smoothening == "situps":
        if hip_angle > 110 and hip_angle<160:
            form = 1
    else:
        form = 1
    return form
def run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle,knee_angle_right ,success_percentage, feedback, workout_name_after_smoothening):
    if workout_name_after_smoothening.strip() == "pushups":
        if form == 1:
            if success_percentage == 0:
                if elbow_angle <= 90 and hip_angle > 160 and elbow_angle_right <= 90 and hip_angle_right > 160:
                    feedback = "Feedback: Go Up"
                    if direction == 0:
                        count["pushups"] += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."
                        
            if success_percentage == 100:
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
                    feedback = "Feedback: Go Down"
                    if direction == 1:
                        count["pushups"] += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
        return [feedback, count["pushups"]]
    # For now, else condition handles just squ`ats
    elif workout_name_after_smoothening.strip() == "squats":
        if form == 1:
            if success_percentage == 0:
                if knee_angle < 90:
                    feedback = "Go Up"
                    if direction == 0:
                        count["squats"] += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."                    
            if success_percentage == 100:
                if knee_angle > 169:
                    feedback = "Feedback: Go Down"
                    if direction == 1:
                        count["squats"] += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
            return [feedback, count["squats"]]
    elif workout_name_after_smoothening.strip() == "bicep_curls":
        if form == 1:
            if success_percentage == 0:
                if elbow_angle < 60:
                    feedback = "Move"
                    if direction == 0:
                        count["bicep_curls"] += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."                    
            if success_percentage == 100:
                if elbow_angle >= 160:
                    feedback = "Release"
                    if direction == 1:
                        count["bicep_curls"] += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
            return [feedback, count["bicep_curls"]]
    elif workout_name_after_smoothening.strip() == "dumbbell_rows":
                if form == 1:
                    if success_percentage == 0:
                        if elbow_angle > 25 and elbow_angle < 40:
                            feedback = "Hands up"
                            if direction == 0:
                                count["dumbbell_rows"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if elbow_angle > 150 and elbow_angle <180:
                            feedback = "Hands Down"
                            if direction == 1:
                                count["dumbbell_rows"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
                    return [feedback, count["dumbbell_rows"]]
    elif workout_name_after_smoothening.strip() == "lunges":
        if form == 1:
            if success_percentage == 0:
                if knee_angle <180 and knee_angle>160:
                    feedback = "Down"
                    if direction == 0:
                        count["lunges"] += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."                    
            if success_percentage == 100:
                if knee_angle <105 and knee_angle>80 :
                    feedback = "Relax"
                    if direction == 1:
                        count["lunges"] += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
            return [feedback, count["lunges"]]
    elif workout_name_after_smoothening.strip() == "situps":
        if form == 1:
            if success_percentage == 0:
                if knee_angle < 65 and knee_angle >30 and hip_angle <130 and hip_angle >90:
                    feedback = "Go Up"
                    if direction == 0:
                        count["situps"] += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."                    
            if success_percentage == 100:
                if knee_angle < 53 and knee_angle >30 and hip_angle <55 and hip_angle >40:
                    feedback = "Go Down"
                    if direction == 1:
                        count["situps"] += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
            return [feedback, count["situps"]]
    elif workout_name_after_smoothening.strip() == "jumping_jacks":
                if form == 1:
                    if success_percentage == 0:
                        if shoulder_angle < 90 and shoulder_angle > 30:
                            feedback = "Hands up"
                            if direction == 0:
                                count["jumping_jacks"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if shoulder_angle > 140 and shoulder_angle <180:
                            feedback = "Hands Down"
                            if direction == 1:
                                count["jumping_jacks"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
                    return [feedback, count["jumping_jacks"]]       
    else:
        return ["Feedback:",0]

#
def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, success_percentage, progress_bar, workout_name_after_smoothening):
    #Draw the pushup progress bar
    # print(count)
    draw_percentage_progress_bar(form, img, success_percentage, progress_bar)

    #Show the rep count
    display_rep_count(count, img)
        
    #Show the pushup feedback 
    show_workout_feedback(feedback, img)
        
    #Show workout name
    show_workout_name_from_model(img, workout_name_after_smoothening)


def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue, clf = set_video_feed_variables()


    a=10
    #Start video feed and run workout
    while cap.isOpened():
        #Getting image from camera 
        ret, img = cap.read() 
        #Getting video dimensions 
        width  = cap.get(3)  
        height = cap.get(4)  
        
        # Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        #Get pose and draw landmarks
        img = get_pose(img, results, False)

        
        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)
        
        #If landmarks exist, get the relevant workout body angles and run workout. The points used are identifiers for specific joints
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle,knee_angle_right = set_body_angles_from_keypoints(get_angle, img, landmark_list)
            
            workout_name_after_smoothening = set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list)    

            workout_name_after_smoothening = workout_name_after_smoothening.replace("Workout Name:", "").strip()
            success_percentage, progress_bar = set_percentage_bar_and_text(elbow_angle, knee_angle,hip_angle,shoulder_angle,workout_name_after_smoothening)
        
                    
            #Is the form correct at the start?
            form = check_form(elbow_angle, shoulder_angle, hip_angle,knee_angle, form, workout_name_after_smoothening)
        
            #Full workout motion
            if workout_name_after_smoothening.strip() == "pushups":
                if form == 1:
                    if success_percentage == 0:
                        if elbow_angle <= 90 and hip_angle > 160 and elbow_angle_right <= 90 and hip_angle_right > 160:
                            feedback = "Feedback: Go Up"
                            if direction == 0:
                                count["pushups"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."
                                
                    if success_percentage == 100:
                        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
                            feedback = "Feedback: Go Down"
                            if direction == 1:
                                count["pushups"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            # For now, else condition handles just squats
            elif workout_name_after_smoothening.strip() == "squats":
                if form == 1:
                    if success_percentage == 0:
                        if knee_angle < 90:
                            feedback = "Go Up"
                            if direction == 0:
                                count["squats"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if knee_angle > 169:
                            feedback = "Feedback: Go Down"
                            if direction == 1:
                                count["squats"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            elif workout_name_after_smoothening.strip() == "bicep_curls":
                if form == 1:
                    if success_percentage == 0:
                        if elbow_angle <60:
                            feedback = "Move"
                            if direction == 0:
                                count["bicep_curls"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if elbow_angle >= 160:
                            feedback = "Release"
                            if direction == 1:
                                count["bicep_curls"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            elif workout_name_after_smoothening.strip() == "lunges":
                if form == 1:
                    if success_percentage == 0:
                        if knee_angle <180 and knee_angle>160:
                            feedback = "Down"
                            if direction == 0:
                                count["lunges"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if knee_angle <105 and knee_angle>80 :
                            feedback = "Relax"
                            if direction == 1:
                                count["lunges"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            elif workout_name_after_smoothening.strip() == "situps":
                if form == 1:
                    if success_percentage == 0:
                        if True:#knee_angle < 65 and knee_angle >30 and hip_angle <130 and hip_angle >90:
                            feedback = "Go Up"
                            if direction == 0:
                                count["situps"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if True:#knee_angle < 53 and knee_angle >30 and hip_angle <55 and hip_angle >20:
                            feedback = "Go Down"
                            if direction == 1:
                                count["situps"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            elif workout_name_after_smoothening.strip() == "jumping_jacks":
                if form == 1:
                    if success_percentage == 0:
                        if shoulder_angle < 90 and shoulder_angle > 30:
                            feedback = "Hands up"
                            if direction == 0:
                                count["jumping_jacks"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if shoulder_angle > 140 and shoulder_angle <180:
                            feedback = "Hands Down"
                            if direction == 1:
                                count["jumping_jacks"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
            elif workout_name_after_smoothening.strip() == "dumbbell_rows":
                if form == 1:
                    if success_percentage == 0:
                        if elbow_angle > 25 and elbow_angle < 40:
                            feedback = "Hands up"
                            if direction == 0:
                                count["dumbbell_rows"] += 0.5
                                direction = 1
                        else:
                            feedback = "Feedback: Bad Form."                    
                    if success_percentage == 100:
                        if elbow_angle > 150 and elbow_angle <180:
                            feedback = "Hands Down"
                            if direction == 1:
                                count["dumbbell_rows"] += 0.5
                                direction = 0
                        else:
                            feedback = "Feedback: Bad Form."
                    # return [feedback, count["bicep_curls"]]
            #Display workout stats  
            # print(workout_name_after_smoothening.strip(),count[workout_name_after_smoothening.strip()])      
            display_workout_stats(count[workout_name_after_smoothening.strip()], form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, success_percentage, progress_bar, workout_name_after_smoothening)
            
            
        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 500, 150
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)      
        alpha = 0.8  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
            
        cv2.imshow('Gym Assistant - Workout Trainer', image_new)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
