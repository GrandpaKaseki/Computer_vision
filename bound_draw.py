import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter.filedialog as fd
import wx


class BoundVIdCV:

    def __init__(self) -> None:
        pass

    def load_file(self, path_to_vid: str):
        """This function takes path to original video"""
        cap = cv2.VideoCapture(path_to_vid)
        if not cap.isOpened():
            print("Error in file reading")
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.cap = cap

    @staticmethod
    def _find_bound_coord(image: np.ndarray) -> dict:
        """this function turn RGB2GRAY, then finding bound coords.
        returns dict, where for each space point key: x_val; value: y_val
        """
        gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # get from RGB to GRAY

        # making black borders white
        for i in range(len(gray_im)):
            for j in range(len(gray_im[i])):
                if gray_im[i, j] < 5:
                    gray_im[i, j] = 255

        # blurred = cv2.GaussianBlur(gray_im, (3, 3), 0)  # some blur to make waves detection more clear
        _, thresh_img = cv2.threshold(gray_im, 50, 255,
                                      cv2.THRESH_BINARY)  # turn all pixels brighter than 50 into white, else into black

        # making easier access to column of pixels. turning picture
        im_copy = np.transpose(thresh_img)
        # creating black picture to draw bound on it
        good_boy = np.zeros_like(im_copy)
        # finding color gradient in each column of pixels
        for column in range(0, len(im_copy)):
            good_boy[column, 0] = good_boy[column, -1] = 0
            for elem in range(1, len(im_copy[column]) - 1):
                if im_copy[column, elem - 1] == im_copy[column, elem] == im_copy[column, elem + 1]:
                    good_boy[column, elem] = 0
                else:
                    good_boy[column, elem] = 255
        # turn picture back to normal
        bound_img = np.transpose(good_boy)

        # get all bound values
        for i in range(len(bound_img[0])):
            bound_img[0][i] = 0
        y_loc, x_loc = np.where(bound_img >= 200)
        # from all y values for each x val  we choose only the most height value
        bound_val = dict()
        for i in range(len(x_loc)):
            if x_loc[i] in bound_val and bound_val[x_loc[i]] > y_loc[i]:
                bound_val[x_loc[i]] = (y_loc[i])
            elif x_loc[i] not in bound_val:
                bound_val[x_loc[i]] = y_loc[i]
        return bound_val

    @staticmethod
    def create_table(self, bound_val: dict, iter_count: int = 0):
        """Creates table which contains data about bound coords"""
        x_val = bound_val.keys()
        x_val = np.sort(list(x_val))
        y_val = [round((self.height - bound_val[x])/self.height, 4) for x in x_val]
        x_val = [round(val/self.width, 4) for val in x_val]
        if iter_count == 0:
            table = pd.DataFrame({f'x, x/x_max': x_val,
                                  f'time_{iter_count}, mls': [self.cap.get(cv2.CAP_PROP_POS_MSEC)] * len(bound_val),
                                  f'y_{iter_count}, y/y_max': y_val})
        else:
            table = pd.DataFrame({f'time_{iter_count}, mls': [self.cap.get(cv2.CAP_PROP_POS_MSEC)] * len(bound_val),
                                  f'y_{iter_count}, y/y_max': y_val})
        table[f'time_{iter_count}, mls'] = table[f'time_{iter_count}, mls'].astype('float64')
        return table

    @staticmethod
    def _draw_points(image: np.ndarray, bound_val: dict) -> None:
        for x in bound_val:
            cv2.circle(image, (x, bound_val[x]), radius=0, color=(0, 255, 255), thickness=2)

    def return_frame(self, time_sec: float = 0, make_table: bool = False, table_name: str = "frame_table") -> None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = self.cap.read()

        if not ret:
            print("Can't receive frame. Maybe video does`t have so much seconds")
        bound_dict = self._find_bound_coord(frame)
        self._draw_points(frame, bound_dict)

        cv2.imshow("output", frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if make_table:
            table = self.create_table(self, bound_dict)
            table.to_csv(table_name + ".csv", sep=";", index=False)

    def return_whole_vid(self, path_to_ret: str, time_sec: float = 0, make_table: bool = False,
                         table_name: str = "video_table") -> None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        output = cv2.VideoWriter(path_to_ret + ".avi", fourcc, self.fps, (int(self.width), int(self.height)))
        frame_counter: int = 0
        frame_skip: int = 2
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"End of the video. Saving result in {path_to_ret}.avi")
                break

            # finding bound only for each frame_skip frame
            if frame_counter % frame_skip == 0:
                bound_dict = self._find_bound_coord(frame)
                self._draw_points(frame, bound_dict)
                if make_table and frame_counter == 0:
                    table = self.create_table(self, bound_dict, frame_counter // frame_skip)
                elif make_table and frame_counter > 0:
                    new_rows = self.create_table(self, bound_dict, frame_counter // frame_skip)
                    table = pd.concat((table, new_rows), axis=1)


            else:
                self._draw_points(frame, bound_dict)

            output.write(frame)
            cv2.imshow("output", frame)
            frame_counter += 1
            if cv2.waitKey(1) & 0xFF == ord('s'):  # press "s" to stop program
                break

        cv2.destroyAllWindows()
        output.release()
        self.cap.release()
        if make_table:
            # print(table.info())
            table.to_csv(table_name + ".csv", sep=";", index=False)

    def draw_bound_anim(self, path_to_table):
        data = pd.read_csv(path_to_table, sep=";")
        x_val = data['x, x/x_max']
        y_val = data['y_0, y/y_msx']
        time = data['time_0, mls'][4]
        fig, ax = plt.subplots()
        plt.title(f"Профиль волны")
        plt.ion()
        line_good, = ax.plot(x_val, y_val, "k", label="Состояние покоя")
        line, = ax.plot(x_val, y_val, label="Профиль волны")
        for i in range(len(list(data))):
            y_val = data[f'y_{i}, y/y_msx']
            line.set_ydata(y_val)
            plt.draw()
            plt.gcf().canvas.flush_events()


vid = BoundVIdCV()
vid.load_file("../video/video_5_new.mp4")
#vid.return_frame(20)
vid.return_whole_vid("../video/arbuz", make_table=True)
# vid.draw_bound_anim("video_table.csv")


