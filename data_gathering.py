import cv2

# Initialize the first camera
cap1 = cv2.VideoCapture(1)  # Replace 0 with the correct ID for your first camera
# Initialize the second camera
cap2 = cv2.VideoCapture(2)  # Replace 1 with the correct ID for your second camera

img_idx = 0

while True:
    # Read from the first camera
    ret1, frame1 = cap1.read()
    # Read from the second camera
    ret2, frame2 = cap2.read()
    
    if ret1:
        # Display the frame from the first camera
        cv2.imshow('Camera left', frame1)
        
    if ret2:
        # Display the frame from the second camera
        cv2.imshow('Camera right', frame2)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        img_idx = img_idx+1
        cv2.imwrite('koon_forward/left/img'+ str(img_idx) +'.png', frame1)
        cv2.imwrite('koon_forward/right/img'+ str(img_idx) +'.png', frame2)
        print("Write Left and Right images")
        print('Images/left/img'+ str(img_idx) +'.png')
        print('Images/right/img'+ str(img_idx) +'.png')
        
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources
cap1.release()
cap2.release()

# Close all OpenCV windows
cv2.destroyAllWindows()