# from TK import loopcap, createUI
# import threading
from GUI import createUI

'''def process(roi):
    roi_blured = Blur(roi)
    roi_skined = skinDetc(roi_blured)
    hand_contour = drawContour(roi_skined)
    return hand_contour


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1: y2, x1: x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    return dst


def image2tensor(image):
    # image = tf.image.decode_jpeg(roi, channels=3)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # image = tf.expand_dims(image, axis=2)
    # print(roi.shape)
    image = tf.image.resize(image, [64, 64])
    image /= 255
    image = tf.expand_dims(image, axis=0)
    return image'''


if __name__ == '__main__':

    '''cap = cv.VideoCapture(0)
    moudel = keras.models.load_model('hand_recognition.h5')

    while True:
        _, frame = cap.read()
        roi = get_roi(frame, 100, 350, 50, 300)
        # cv.imshow('roi', roi)

        roi = process(roi)
        roi_tensor = image2tensor(roi)
        # print(roi.shape)
        prediction = moudel.predict(roi_tensor)
        result = np.argmax(prediction, axis=1)
        prob = prediction[0]
        print(prob[0])

        results = ['ROCK', 'PAPER', 'SCISSORS']
        print(results[result[0]])
        text = 'result is ' + results[result[0]]
        cv.putText(img=frame, text=text, org=(100, 50),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255))

        cv.imshow('frame', frame)
        cv.imshow('roi', roi)
        k = cv.waitKey(50)
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()'''
    '''t = threading.Thread(target=loopcap)
    t.setDaemon(True)
    t.start()
    createUI()'''

    createUI()
