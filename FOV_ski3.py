# This program overlays another camera's FOV on a pre-recorded video

import csv
import numpy as np
import cv2 as cv

# FOV variables from cameras
phone_fovx = 63.5
phone_fovy = 35.7
camera_fovx = 20
camera_fovy = 11.3
centerdiff_x = 100
centerdiff_y = 80
camera_pixelx = 1280
camera_pixely = 720

cap = cv.VideoCapture('Pictures/Ski1.mp4') # video size is 1080x1920
watermark = cv.imread('Pictures/mapicon.png', -1)
watermark_h, watermark_w, watermark_c = watermark.shape # watermark_h = 188, watermark_w = 166
watermark1 = cv.resize(watermark, (int(watermark_w * 1/3), int(watermark_h * 1/3)), interpolation=cv.INTER_AREA)
watermark1_h, watermark1_w, watermark1_c = watermark1.shape # watermark1_h = 62, watermark1_w = 55
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('FOV_ski3.avi',fourcc, 30.0, (1920,1080))
frame_counter = 0

# take first frame of the video
ret, frame = cap.read()
frame_h, frame_w, frame_c = frame.shape

# ratio calculation
x_ratio = (camera_fovx/camera_pixelx) * (frame_w/phone_fovx)
y_ratio = (camera_fovy/camera_pixely) * (frame_h/phone_fovy)

# resize watermark
watermark_fixed_w = int(watermark1_w * (camera_fovx / camera_pixelx) * (frame_w / phone_fovx))
watermark_fixed_h = int(watermark1_h * (camera_fovy / camera_pixely) * (frame_h / phone_fovy))
watermark_fixed = cv.resize(watermark1, (watermark_fixed_w, watermark_fixed_h))

# read csv file
with open('Pictures/Ski1.csv') as f:
    filereader = csv.reader(f, delimiter=",")
    x_coord = []
    y_coord = []
    f.readline()
    for row in filereader:
        if row[1] == "None":
            x_coord.append("None")
        else:
            x_coord.append(int(row[1]))
        if row[2] == "None":
            y_coord.append("None")
        else:
            y_coord.append(int(row[2]))

while cap.isOpened():
    frame_counter += 1
    ret, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape

    # calculate phone fov center location by x, y coordinates
    phone_centerx = int(frame_w/2)
    phone_centery = int(frame_h/2)

    # draw a blue cross on phone center location
    cv.line(frame, (phone_centerx - 10, phone_centery), (phone_centerx + 10, phone_centery), (255, 0, 0), 2)
    cv.line(frame, (phone_centerx, phone_centery - 10), (phone_centerx, phone_centery + 10), (255, 0, 0), 2)

    # calculate camera center location by x, y coordinates
    camera_centerx = phone_centerx + centerdiff_x
    camera_centery = phone_centery + centerdiff_y

    # draw a green cross on camera center location
    cv.line(frame, (camera_centerx - 10, camera_centery), (camera_centerx + 10, camera_centery), (0, 255, 0), 2)
    cv.line(frame, (camera_centerx, camera_centery - 10), (camera_centerx, camera_centery + 10), (0, 255, 0), 2)

    # add camera fov green rectangle
    # calculate rectangle x, y coordinates
    camera_xsize = int(camera_fovx/phone_fovx * frame_w)
    camera_ysize = int(camera_fovy/phone_fovy * frame_h)
    camera_x = camera_centerx - int(camera_xsize/2)
    camera_y = camera_centery - int(camera_ysize/2)

    cv.rectangle(frame, (camera_x, camera_y), (camera_x + camera_xsize, camera_y + camera_ysize), (0, 255, 0), 3)
    #cv.circle(frame, (camera_x, camera_y), 10, (0, 0, 255), -1)

    # add frame count to video
    text_color = (255, 0, 0)  # color as (B,G,R)
    cv.putText(frame, "Frame count: " + str(frame_counter), (50, 50), cv.FONT_HERSHEY_PLAIN, 2.0, text_color, thickness=3, lineType=8)

    # overlay watermark to frame
    # fix x_coord and y_coord lists to add the right watermark location to fit camera fov
    if (x_coord[frame_counter - 1] != "None") and (y_coord[frame_counter - 1] != "None") and (x_coord[frame_counter - 1] >= 0) and (y_coord[frame_counter - 1] >= 0):
        x_coord[frame_counter - 1] = int(x_coord[frame_counter - 1] * x_ratio) + camera_x
        y_coord[frame_counter - 1] = int(y_coord[frame_counter - 1] * y_ratio) + camera_y

        overlay1 = np.zeros((frame_h, frame_w, 4), dtype = 'uint8')
        overlay1[y_coord[frame_counter-1]: y_coord[frame_counter-1] + watermark_fixed_h, x_coord[frame_counter-1]: x_coord[frame_counter-1] + watermark_fixed_w] = watermark_fixed[0:watermark_fixed_h, 0:watermark_fixed_w]
        cv.addWeighted(overlay1, 0.5, frame, 1, 0, frame)

    print(frame_counter, x_coord[frame_counter - 1], y_coord[frame_counter - 1])

    frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(17) & 0xFF == 27:
        break

cap.release()
out.release()
cv.destroyAllWindows()