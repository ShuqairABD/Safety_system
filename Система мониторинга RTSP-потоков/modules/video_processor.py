import logging
import threading
import cv2
import os
import queue
import sqlite3
import shutil
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime
import pytz
import time
from telegram import Bot
from ultralytics import YOLO
import zipfile
from io import BytesIO
import pandas as pd
from openpyxl import Workbook




logging.basicConfig(level=logging.INFO)

class HelperFunctions:
    @staticmethod
    def non_max_suppression(boxes, scores, threshold=0.7):
        if len(boxes) == 0:
            return []
        
        x1 = np.array([box[0] for box in boxes])
        y1 = np.array([box[1] for box in boxes])
        x2 = np.array([box[2] for box in boxes])
        y2 = np.array([box[3] for box in boxes])
        
        area = (x2 - x1) * (y2 - y1)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (area[i] + area[indices[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            indices = indices[inds + 1]
            
        return keep

class DeepSort:
    def __init__(self, max_age=10, nms_max_overlap=0.1, max_cosine_distance=0.2, nn_budget=20):
        self.max_age = max_age
        self.nms_max_overlap = nms_max_overlap
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.tracks = []
        self.next_id = 1
        
    def update_tracks(self, detections, frame=None):
        if frame is None:
            return []  # Или self.tracks# Простая имплементация без фактического DeepSORT
        updated_tracks = []
        
        # Обновление существующих треков
        for track in self.tracks:
            track.age += 1
            if track.age > self.max_age:
                continue  # Пропуск слишком старые треки
            updated_tracks.append(track)
        
        # Создание новых треков из детекций
        for bbox, confidence, class_name in detections:
            new_track = Track(self.next_id, bbox, confidence, class_name)
            updated_tracks.append(new_track)
            self.next_id += 1
        
        self.tracks = updated_tracks
        return self.tracks

class Track:
    def __init__(self, track_id, bbox, confidence, class_name):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, width, height]
        self.age = 0
        self.confidence = confidence
        self.det_class = class_name
        
    def is_confirmed(self):
        return True  #  все треки считаются подтвержденными
        
    def to_ltrb(self):
        x, y, w, h = self.bbox
        return [x, y, x + w, y + h]
        
    def get_det_conf(self):
        return self.confidence

class VideoProcessor:
    def __init__(self, DB_PATH):
        self.DB_PATH = DB_PATH
        self.stop_flag = False
        self.streams = {}
        self.lock = threading.Lock()
        self.saved_track_ids = set()
        self.save_lock = threading.Lock()
        self.violation_queue = queue.Queue()  # Очередь для нарушений
        self.sent_violations = {}  # Словарь для отслеживания отправленных нарушений
        self.VIOLATION_COOLDOWN = 180 
        
        self.active_streams = {}  # To track active streams
        
        # Запуск потока для сохранения из очереди
        self.save_thread = threading.Thread(target=self._save_violations_from_queue)
        self.save_thread.daemon = True
        self.save_thread.start()
        
        # Инициализация потока для Telegram
        self.telegram_queue = queue.Queue()
        self.telegram_thread = threading.Thread(target=self._process_telegram_queue)
        self.telegram_thread.daemon = True
        self.telegram_thread.start()

    def process_stream(self, rtsp_url, stream_id):
        self.active_streams[stream_id] = {
            'running': True,
            'thread': threading.current_thread(),
            'url': rtsp_url
        }
        
        # Основной метод обработки видеопотока
        cap = cv2.VideoCapture(rtsp_url)
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Поток {stream_id}: кадр не получен.")
                break

            with self.lock:
                # Важно! Добавить кадр в streams, не перезаписывая весь словарь
                self.streams[stream_id] = frame.copy()

        cap.release()
        logging.info(f"Поток {stream_id} остановлен.")

    def get_processed_frame(self, stream_id):
        with self.lock:
            return self.streams.get(stream_id, None)

    
    def _save_violations_from_queue(self):
        """Сохраняет кадры из очереди в отдельном потоке."""
        while not self.stop_flag or not self.violation_queue.empty():
            try:
                stream_id, track_id, frame, violation_type, local_time = self.violation_queue.get(timeout=1)

                with self.save_lock:
                    violation_frame_path = os.path.join("violation_frame", f"{stream_id}_violation_{track_id}.jpg")
                    success = cv2.imwrite(violation_frame_path, frame)

                    if success:
                        logging.info(f"Сохранён кадр из очереди: {violation_frame_path}")
                    else:
                        logging.error(f"Ошибка сохранения кадра: {violation_frame_path}")

                    conn = sqlite3.connect(self.DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO violations (track_id, violation_type, timestamp) VALUES (?, ?, ?)",
                        (track_id, violation_type, local_time),
                    )
                    conn.commit()
                    conn.close()

                self.violation_queue.task_done()

            except queue.Empty:
                continue

            except Exception as e:
                logging.error(f"Ошибка в сохранении из очереди: {e}")



    def _save_violation_frame(self, stream_id, track_id, processed_frame):  # Используется processed_frame!
        """Сохраняет кадр с bbox'ами"""
        try:
            os.makedirs("violation_frame", exist_ok=True)
            img_path = os.path.join("violation_frame", f"{track_id}_violation.jpg")
            
            # Сохранить КАК ЕСТЬ (с рамками)
            cv2.imwrite(img_path, processed_frame)  # Не RGB!
            
            logging.info(f"Сохранено нарушение с bbox: {img_path}")
        except Exception as e:
            logging.error(f"Ошибка сохранения: {str(e)}")
            
        

    
    def _process_telegram_queue(self):
        """Отправляет фотографии нарушений в Telegram в реальном времени"""
        bot = None
        CHAT_ID = "6381386509"
        
        # Сохранить отправленные сообщения для предотвращения дубликатов
        sent_messages = set()
        
        while not self.stop_flag or not self.telegram_queue.empty():
            try:
                stream_id, frame, violation_type, local_time = self.telegram_queue.get(timeout=1)
                
                # Создать уникальный ключ для этого нарушения
                message_key = f"{stream_id}_{violation_type}_{local_time}"
                
                # Проверить, не отправляли ли уже это сообщение
                if message_key in sent_messages:
                    self.telegram_queue.task_done()
                    continue
                    
                # Добавить ключ в список отправленных
                sent_messages.add(message_key)
                
                # Инициализация бота при первом использовании
                if bot is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    bot = Bot(token="7815016234:AAHco5bJKlvB0qJMyF-d6m0oCIv-msVN7bs")
                
                # Создать сообщение с информацией о нарушении и камере
                async def send_violation():
                    message = f"⚠️ Нарушение: {violation_type}\n"
                    message += f"📷 Камера: {stream_id}\n"
                    message += f"🕒 Время: {local_time}"
                    
                    # Сохранить временный файл
                    temp_file = f"temp_{stream_id}_{int(time.time())}.jpg"
                    cv2.imwrite(temp_file, frame)
                    
                    with open(temp_file, "rb") as photo:
                        await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
                    
                    # Удалять временный файл
                    os.remove(temp_file)
                    
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_violation())
                
                self.telegram_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Ошибка при отправке в Telegram: {e}")
        
    def get_active_streams(self):
        return {stream_id: data['url'] for stream_id, data in self.active_streams.items()}
    
    def stop_single_stream(self, stream_id):
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['running'] = False
            time.sleep(0.5)  # Give thread time to stop
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    def get_processed_frame(self, stream_id):
        """Возвращает текущий кадр из выбранного потока"""
        with self.lock:
            return self.streams.get(stream_id, None)

    def stop_processing(self):
        """Окончательная рабочая версия"""
        
        self.stop_flag = True  # Остановка всех потоков

        try:
            # --- 1. Получить данные из БД ---
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            
            # Читать нарушения
            cursor.execute("SELECT track_id, violation_type, timestamp FROM violations ORDER BY timestamp DESC")
            violations = cursor.fetchall()
            
            # Создать DataFrame
            violation_table = pd.DataFrame(violations, columns=["ID", "Тип нарушения", "Время"])
            
            # --- 2. Сохранить в Excel ---
            excel_path = "violations_report.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Violations"
            ws.append(["ID", "Тип нарушения", "Время"])
            
            for _, row in violation_table.iterrows():
                ws.append([str(row['ID']), str(row['Тип нарушения']), str(row['Время'])])
            
            wb.save(excel_path)

            # --- 3. Подготовка фото с bboxes ---
            violation_images = []
            
            #  кадры напрямую из streams (где они с bboxes)
            with self.lock:  # Важно для потокобезопасности!
                for stream_id, frame in self.streams.items():
                    if frame is not None:
                        # Сохранить оригинальный кадр с bboxes
                        img_path = os.path.join("violation_frame", f"{stream_id}_violation.jpg")
                        cv2.imwrite(img_path, frame)
                        
                        #  для Gradio (BGR ---> RGB)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        violation_images.append(img_rgb)
            
            # --- 4.  статусное сообщение ---
            status_msg = (
                f"✅ Обработка завершена\n"
                f"✅ Данные сохранены в unique_violations.xlsx\n"
                f"✅ Папка архивирована в violation_frames.zip\n"
            )

            # --- 5. Визуальная проверка (можно удалить после отладки) ---
            if violation_images:
                debug_img = cv2.cvtColor(violation_images[-1], cv2.COLOR_RGB2BGR)
                cv2.imshow("DEBUG - Последний кадр", debug_img)
                cv2.waitKey(2000)  # Показ 2 секунды
                cv2.destroyAllWindows()
                print(f"Отладка: размер {debug_img.shape}, bboxes видны: {'красные' in str(debug_img)}")

            # --- 6. Возврат результатов ---
            return status_msg, violation_images, violation_table

        except Exception as e:
            logging.error(f"Ошибка в stop_processing: {str(e)}")
            return (
                f"❌ Критическая ошибка: {str(e)}",
                [],
                pd.DataFrame(columns=["ID", "Тип нарушения", "Время"])
            )

        finally:
            if 'conn' in locals():
                conn.close()
    
   
    def get_violations_from_zip(self):
        """Безопасное получение данных о нарушениях из архива"""
        
        # Инициализация
        violation_images = []
        violation_info_list = []
        empty_df = pd.DataFrame(columns=["ID", "Тип нарушения", "Время"])

        try:
            # 1. Проверка и чтение Excel
            if not os.path.exists("unique_violations.xlsx"):
                logging.error("Excel файл не найден")
                return empty_df, [], "❌ Файл с данными не найден"

            try:
                excel_data = pd.read_excel("unique_violations.xlsx")
            except Exception as e:
                logging.error(f"Ошибка чтения Excel: {str(e)}")
                return empty_df, [], f"❌ Ошибка чтения данных: {str(e)}"

            if excel_data.empty:
                return empty_df, [], "❌ Нет данных о нарушениях"

            # 2. Подготовка табличных данных
            try:
                table_data = excel_data[['track_id', 'violation_type', 'timestamp']].copy()
                table_data.columns = ["ID", "Тип нарушения", "Время"]
            except KeyError as e:
                logging.error(f"Отсутствуют необходимые колонки: {str(e)}")
                return empty_df, [], f"❌ Неправильный формат данных: {str(e)}"

            # 3. Проверка и чтение архива
            if not os.path.exists("violation_frames.zip"):
                logging.warning("Архив с фотографиями не найден")
                return table_data, [], "⚠️ Архив с фотографиями не найден"

            try:
                with zipfile.ZipFile("violation_frames.zip", 'r') as zip_ref:
                    # Получать список файлов в архиве
                    zip_files = zip_ref.namelist()
                    
                    for _, row in excel_data.iterrows():
                        track_id = row['track_id']
                        img_name = f"{track_id}_violation.jpg"
                        
                        if img_name in zip_files:
                            try:
                                with zip_ref.open(img_name) as img_file:
                                    img_data = img_file.read()
                                    nparr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        violation_images.append(img_rgb)
                                        violation_info_list.append(
                                            f"ID: {track_id}, Тип: {row['violation_type']}, "
                                            f"Время: {row['timestamp']}"
                                        )
                                    else:
                                        logging.warning(f"Не удалось декодировать: {img_name}")
                            except Exception as e:
                                logging.error(f"Ошибка чтения {img_name}: {str(e)}")
                        else:
                            logging.warning(f"Фото не найдено в архиве: {img_name}")

            except zipfile.BadZipFile:
                logging.error("Архив повреждён или не является ZIP-файлом")
                return table_data, [], "❌ Ошибка: архив повреждён"
            
            # 4. Формирование результатов
            violation_info = "\n".join(violation_info_list) if violation_info_list else "Нет данных"
            return table_data, violation_images, violation_info

        except Exception as e:
            logging.error(f"Критическая ошибка: {str(e)}")
            return empty_df, [], f"❌ Системная ошибка: {str(e)}"
        

    def get_violation_data_for_display(self):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            query = """
            SELECT track_id, violation_type, timestamp
            FROM violations
            ORDER BY timestamp DESC
            """
            data = pd.read_sql_query(query, conn)
            conn.close()

            if data.empty:
                return [], "❌ Нет данных о нарушениях."

            violation_images = []
            violation_info_list = []
            base_path = os.path.abspath("violation_frame")  # Полный абсолютный путь

            for _, row in data.iterrows():
                track_id = row['track_id']
                violation_type = row['violation_type']
                timestamp = row['timestamp']
                image_path = os.path.join(base_path, f"{track_id}_violation.jpg")

                if os.path.exists(image_path):
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Конвертация в RGB
                            violation_images.append(image_rgb)
                            violation_info_list.append(
                                f"**ID:** {track_id}, **Тип:** {violation_type}, **Время:** {timestamp}"
                            )
                        else:
                            logging.error(f"Ошибка: файл {image_path} не является изображением")
                    except Exception as e:
                        logging.error(f"Ошибка при загрузке {image_path}: {e}")
                else:
                    logging.error(f"Файл не найден: {image_path}")

            violation_info = "\n\n".join(violation_info_list)
            return violation_images, violation_info

        except Exception as e:
            logging.error(f"Ошибка в get_violation_data_for_display: {e}")
            return [], f"❌ Ошибка: {e}"


    def process_video(
        self,
        conf_threshold,
        iou_threshold,
        max_age_minutes=1,
        nms_max_overlap=0.2,
        max_cosine_distance=0.2,
        nn_budget=10,
        frame_skip_ratio=0.7,
        max_det=5,
        show_track_ids=True,
        show_confidence=True,
        max_distance=30,
        max_size_change=30,
    ):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM violations")
            conn.commit()
            print("✅ База данных очищена.")

            if not os.path.exists("violation_frame"):
                os.makedirs("violation_frame")
            for filename in os.listdir("violation_frame"):
                file_path = os.path.join("violation_frame", filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")
            print("✅ Папка 'violation_frame' очищена.")
            
            model = YOLO("best_yolov8s.pt")
          # model.export(format="openvino", int8=True)
          # model.export(format="onnx", half=True)
            if model is None:
                return "Ошибка: модель YOLO не загружена.", None, None
            
            

            fps = 30
            frame_skip = int(fps * frame_skip_ratio) if frame_skip_ratio is not None else 21  # Default to 21 if None
            
            #  словарь для отслеживания кадров для каждого потока
            frame_count = {}

            # Use default values if parameters are None
            max_age = int((max_age_minutes if max_age_minutes is not None else 1) * 60 * fps)
            nms_max_overlap = nms_max_overlap if nms_max_overlap is not None else 0.2
            max_cosine_distance = max_cosine_distance if max_cosine_distance is not None else 0.2
            nn_budget = nn_budget if nn_budget is not None else 10

            tracker = DeepSort(
                max_age=max_age,
                nms_max_overlap=nms_max_overlap,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
            )
            class_names = {0: "Helmet", 1: "No-Helmet", 2: "No-Vest", 3: "Vest"}
            
            # Также инициализация словарь для отслеживания предыдущих треков для каждого потока
            previous_tracks = {}

            while not self.stop_flag:
                frames_to_process = []
                with self.lock:
                    frames_to_process = list(self.streams.items())

                if not frames_to_process:
                    time.sleep(0.1)
                    continue

                for stream_id, frame in frames_to_process:
                    if frame is None:
                        logging.warning(f"Кадр для потока {stream_id} не получен")
                        continue

                    # Инициализация счетчика для нового потока
                    if stream_id not in frame_count:
                        frame_count[stream_id] = 0
                    
                    # Инициализация словаря треков для нового потока
                    if stream_id not in previous_tracks:
                        previous_tracks[stream_id] = {}

                    # Увеличивать счетчик для этого конкретного потока
                    frame_count[stream_id] += 1
                    current_frame_count = frame_count[stream_id]
                    
                    print(f"Обработка кадра {current_frame_count} для потока {stream_id}")
                    
                    # Пропус кадры в соответствии с настройкой frame_skip
                    if frame_skip > 0 and current_frame_count % frame_skip != 0:
                        continue
                    
                    # Safe YOLO inference
                    try:
                        results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
                        if results is None or len(results.boxes) == 0:
                            print(f"YOLO не обнаружил объекты на этом кадре для потока {stream_id}.")
                            continue
                    except Exception as e:
                        logging.error(f"Ошибка при выполнении YOLO: {e}")
                        continue

                    all_boxes = []
                    all_scores = []
                    detections = []
                    has_violation = False

                    # детекции - с проверками на None
                    for box in results.boxes:
                        try:
                            if box.xyxy is None or box.cls is None or box.conf is None:
                                continue
                                
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            confidence = box.conf[0].item()
                            label = class_names.get(class_id, "Unknown")
                            
                            all_boxes.append([x1, y1, x2, y2])
                            all_scores.append(confidence)
                            logging.info(f"Детекция: {label} на {stream_id} с уверенностью {confidence}")
                            
                            if label in class_names.values():
                                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, label))
                            if label in ["No-Helmet", "No-Vest"]:
                                has_violation = True
                        except Exception as e:
                            logging.error(f"Ошибка при обработке бокса: {e}")
                            continue

                    # Безопасный NMS с проверкой на пустые списки
                    if not all_boxes or not all_scores:
                        continue
                        
                    try:
                        nms_indices = HelperFunctions.non_max_suppression(all_boxes, all_scores, threshold=iou_threshold)
                        filtered_boxes = [all_boxes[i] for i in nms_indices]
                        if max_det is not None and len(filtered_boxes) > max_det:
                            filtered_boxes = filtered_boxes[:max_det]
                    except Exception as e:
                        logging.error(f"Ошибка в NMS: {e}")
                        continue

                    # Обработка кадра для отображения
                    processed_frame = frame.copy()

                    # Обновлять трекинг с проверками
                    try:
                        tracks = tracker.update_tracks(detections, frame=processed_frame)
                    except Exception as e:
                        logging.error(f"Ошибка в обновлении треков: {e}")
                        continue
                        
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                            
                        try:
                            track_id = track.track_id
                            x1, y1, x2, y2 = map(int, track.to_ltrb())
                            label = track.det_class if track.det_class else "Unknown"

                            # Проверка предыдущих треков для конкретного потока
                            if track_id in previous_tracks[stream_id]:
                                prev_x1, prev_y1, prev_x2, prev_y2 = previous_tracks[stream_id][track_id]
                                
                                # Safe distance calculation with None checks
                                if (max_distance is not None and 
                                    not (None in [x1, prev_x1, y1, prev_y1])):
                                    distance = np.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)
                                    
                                    if (max_size_change is not None and 
                                        not (None in [x2, x1, prev_x2, prev_x1, y2, y1, prev_y2, prev_y1])):
                                        size_change = abs((x2 - x1) - (prev_x2 - prev_x1)) + abs((y2 - y1) - (prev_y2 - prev_y1))
                                        
                                        if distance > max_distance or size_change > max_size_change:
                                            continue
                                    else:
                                        if distance > max_distance:
                                            continue
                                            
                            previous_tracks[stream_id][track_id] = (x1, y1, x2, y2)

                            color = (0, 0, 255) if label in ["No-Helmet", "No-Vest"] else (0, 255, 0)
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                            
                            if show_track_ids:
                                cv2.putText(processed_frame, f"ID: {track_id}", (x1, y1 - 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                           
                            if show_confidence:
                                confidence = track.get_det_conf() if track.get_det_conf() is not None else 0
                                cv2.putText(processed_frame, f"Conf: {confidence:.2f}", (x1, y1 - 70), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                           
                            cv2.putText(processed_frame, label, (x1, y1 - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        except Exception as e:
                            logging.error(f"Ошибка при обработке трека {track_id if 'track_id' in locals() else 'unknown'}: {e}")
                            continue

                    # Обновление кадра в streams для отображения на интерфейсе
                    with self.lock:
                        self.streams[stream_id] = processed_frame

                    # Если есть нарушение и прошла проверка, отправляется кадр в очередь
                    if has_violation:
                        try:
                            violation_type = "No-Helmet" if "No-Helmet" in [t.det_class for t in tracks if t.det_class] else "No-Vest"
                            local_time = datetime.now(pytz.timezone("Europe/Moscow")).strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Отфильтровать только треки с нарушениями
                            violation_tracks = [track for track in tracks if track.det_class in ["No-Helmet", "No-Vest"]]
                            
                            # Создать копию кадра для отметок
                            violation_frame = frame.copy()
                            
                            # Отрисовка только bboxes нарушителей
                            for track in violation_tracks:
                                try:
                                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                                    track_id = track.track_id
                                    label = track.det_class
                                    confidence = track.get_det_conf() if track.get_det_conf() is not None else 0
                                    
                                    # Проверка уникальности нарушения
                                    current_time = time.time()
                                    violation_key = (stream_id, track_id, label)
                                    
                                    if violation_key in self.sent_violations:
                                        last_sent_time = self.sent_violations[violation_key]
                                        if current_time - last_sent_time < self.VIOLATION_COOLDOWN:
                                            continue
                                    
                                    # Обновлять словарь отправленных нарушений
                                    self.sent_violations[violation_key] = current_time
                                    
                                    # Очистка старых записей
                                    self.sent_violations = {
                                        k: v for k, v in self.sent_violations.items() 
                                        if current_time - v < self.VIOLATION_COOLDOWN
                                    }
                                    
                                    # bbox только для нарушителей
                                    color = (0, 0, 255)  # red
                                    cv2.rectangle(violation_frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(violation_frame, f"ID: {track_id}", (x1, y1 - 50), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                    cv2.putText(violation_frame, f"Conf: {confidence:.2f}", (x1, y1 - 70), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                    cv2.putText(violation_frame, label, (x1, y1 - 30), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                    
                                    # Отправка в очереди только текущего нарушения
                                    self.violation_queue.put((stream_id, track_id, violation_frame, label, local_time))
                                    self.telegram_queue.put((stream_id, violation_frame, label, local_time))
                                    
                                except Exception as e:
                                    logging.error(f"Ошибка при обработке нарушения для трека {track_id if 'track_id' in locals() else 'unknown'}: {e}")
                        except Exception as e:
                            logging.error(f"Ошибка при обработке нарушений: {e}")

            # Сохранить только уникальные нарушения в Excel
            query = """
            SELECT DISTINCT track_id, violation_type, MIN(timestamp) AS first_occurrence
            FROM violations
            GROUP BY track_id, violation_type
            ORDER BY first_occurrence
            """
            data = pd.read_sql_query(query, conn)
            excel_path = "unique_violations.xlsx"
            data.to_excel(excel_path, index=False)
            logging.info(f"✅ Данные сохранены в {excel_path}")
            

            zip_path = "violation_frames.zip"
            shutil.make_archive("violation_frames", 'zip', "violation_frame")
            logging.info(f"✅ Папка с фотографиями архивирована в {zip_path}")
            print (f"✅ Папка с фотографиями архивирована в {zip_path}")

            conn.close()
            return "✅ Обработка завершена, Все данные сохранены и готовый для скачки", excel_path, zip_path
        
        except Exception as e:
            if 'conn' in locals():
                conn.close()
            logging.error(f"Ошибка при обработке: {e}")
            return f"Ошибка при обработке: {str(e)}", None, None

    def process_video(
        self,
        conf_threshold,
        iou_threshold,
        max_age_minutes=1,
        nms_max_overlap=0.2,
        max_cosine_distance=0.2,
        nn_budget=10,
        frame_skip_ratio=0.7,
        max_det=5,
        show_track_ids=True,
        show_confidence=True,
        max_distance=30,
        max_size_change=30,
    ):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM violations")
            conn.commit()
            print("✅ База данных очищена.")

            if not os.path.exists("violation_frame"):
                os.makedirs("violation_frame")
            for filename in os.listdir("violation_frame"):
                file_path = os.path.join("violation_frame", filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")
            print("✅ Папка 'violation_frame' очищена.")

            model = YOLO("best_yolov8s.pt")
            if model is None:
                return "Ошибка: модель YOLO не загружена.", None, None

            fps = 30
            frame_skip = int(fps * frame_skip_ratio) if frame_skip_ratio is not None else 21
            
            # отдельные счетчики кадров для каждого потока
            frame_counts = {}
            
            # Отдельные трекеры для каждого потока
            trackers = {}
            previous_tracks = {}
            
            class_names = {0: "Helmet", 1: "No-Helmet", 2: "No-Vest", 3: "Vest"}

            while not self.stop_flag:
                frames_to_process = []
                with self.lock:
                    frames_to_process = list(self.streams.items())

                if not frames_to_process:
                    time.sleep(0.1)  # Небольшая задержка чтобы не нагружать CPU
                    continue

                for stream_id, frame in frames_to_process:
                    if frame is None:
                        logging.warning(f"Кадр для потока {stream_id} не получен")
                        continue
                    
                    # Инициализация счетчика кадров для этого потока, если его еще нет
                    if stream_id not in frame_counts:
                        frame_counts[stream_id] = 0
                        
                    # Инициализация трекера для этого потока, если его еще нет
                    if stream_id not in trackers:
                        trackers[stream_id] = DeepSort(
                            max_age=int(max_age_minutes * 60 * fps),
                            nms_max_overlap=nms_max_overlap,
                            max_cosine_distance=max_cosine_distance,
                            nn_budget=nn_budget,
                        )
                    
                    # Инициализация списка предыдущих треков для этого потока
                    if stream_id not in previous_tracks:
                        previous_tracks[stream_id] = {}

                    # Увеличить счетчик для этого конкретного потока
                    frame_counts[stream_id] += 1
                    current_frame_count = frame_counts[stream_id]
                    
                    # Пропус кадры в соответствии с настройкой frame_skip
                    if current_frame_count % frame_skip != 0:
                        continue

                    print(f"Обработка кадра {current_frame_count} для потока {stream_id}")
                    
                    try:
                        results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
                        if results is None or len(results.boxes) == 0:
                            print(f"YOLO не обнаружил объекты на этом кадре для потока {stream_id}.")
                            continue
                            
                        all_boxes = []
                        all_scores = []
                        detections = []
                        has_violation = False

                        # детекции
                        for box in results.boxes:
                            try:
                                if box.xyxy is None or box.cls is None or box.conf is None:
                                    continue
                                    
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                class_id = int(box.cls[0])
                                confidence = box.conf[0].item()
                                label = class_names.get(class_id, "Unknown")
                                
                                all_boxes.append([x1, y1, x2, y2])
                                all_scores.append(confidence)
                                logging.info(f"Детекция: {label} на {stream_id} с уверенностью {confidence}")
                                
                                if label in class_names.values():
                                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, label))
                                if label in ["No-Helmet", "No-Vest"]:
                                    has_violation = True
                            except Exception as e:
                                logging.error(f"Ошибка при обработке бокса: {e}")
                                continue

                        # NMS с проверкой на пустые списки
                        if not all_boxes or not all_scores:
                            continue
                            
                        try:
                            nms_indices = HelperFunctions.non_max_suppression(all_boxes, all_scores, threshold=iou_threshold)
                            filtered_boxes = [all_boxes[i] for i in nms_indices]
                            if max_det is not None and len(filtered_boxes) > max_det:
                                filtered_boxes = filtered_boxes[:max_det]
                        except Exception as e:
                            logging.error(f"Ошибка в NMS: {e}")
                            continue

                        # Создать копию кадра для обработки
                        processed_frame = frame.copy()

                        # Обновлять трекинг с использованием трекера для этого потока
                        try:
                            tracks = trackers[stream_id].update_tracks(detections, frame=processed_frame)
                        except Exception as e:
                            logging.error(f"Ошибка в обновлении треков для потока {stream_id}: {e}")
                            continue

                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                                
                            try:
                                track_id = track.track_id
                                x1, y1, x2, y2 = map(int, track.to_ltrb())
                                label = track.det_class if track.det_class else "Unknown"

                                # Проверка треков с учетом stream_id
                                if track_id in previous_tracks[stream_id]:
                                    prev_x1, prev_y1, prev_x2, prev_y2 = previous_tracks[stream_id][track_id]
                                    
                                    if max_distance is not None:
                                        distance = np.sqrt((x1 - prev_x1) ** 2 + (y1 - prev_y1) ** 2)
                                        
                                        if max_size_change is not None:
                                            size_change = abs((x2 - x1) - (prev_x2 - prev_x1)) + abs((y2 - y1) - (prev_y2 - prev_y1))
                                            
                                            if distance > max_distance or size_change > max_size_change:
                                                continue
                                        else:
                                            if distance > max_distance:
                                                continue

                                # Обновлять предыдущие треки для этого потока
                                previous_tracks[stream_id][track_id] = (x1, y1, x2, y2)

                                color = (0, 0, 255) if label in ["No-Helmet", "No-Vest"] else (0, 255, 0)
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                                
                                if show_track_ids:
                                    cv2.putText(processed_frame, f"ID: {track_id}", (x1, y1 - 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                            
                                if show_confidence:
                                    confidence = track.get_det_conf() if track.get_det_conf() is not None else 0
                                    cv2.putText(processed_frame, f"Conf: {confidence:.2f}", (x1, y1 - 70), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                            
                                cv2.putText(processed_frame, label, (x1, y1 - 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            except Exception as e:
                                logging.error(f"Ошибка при обработке трека {track_id if 'track_id' in locals() else 'unknown'} в потоке {stream_id}: {e}")
                                continue

                        # Обновлять кадр в streams для отображения на интерфейсе
                        with self.lock:
                            self.streams[stream_id] = processed_frame

                        # Если есть нарушение, отправлять кадр в очередь
                        if has_violation:
                            try:
                                violation_type = "No-Helmet" if "No-Helmet" in [t.det_class for t in tracks if t.det_class] else "No-Vest"
                                local_time = datetime.now(pytz.timezone("Europe/Moscow")).strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Отфильтровываем треки с нарушениями
                                violation_tracks = [track for track in tracks if track.det_class in ["No-Helmet", "No-Vest"]]
                                
                                # Создать копию кадра для отметок нарушений
                                violation_frame = frame.copy()
                                
                                # Отрисовывать только bboxes нарушителей
                                for track in violation_tracks:
                                    try:
                                        track_id = track.track_id
                                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                                        label = track.det_class
                                        confidence = track.get_det_conf() if track.get_det_conf() is not None else 0
                                        
                                        # Рамка красного цвета для нарушителей
                                        color = (0, 0, 255)  # красный
                                        cv2.rectangle(violation_frame, (x1, y1), (x2, y2), color, 2)
                                        cv2.putText(violation_frame, f"ID: {track_id}", (x1, y1 - 50), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                        cv2.putText(violation_frame, f"Conf: {confidence:.2f}", (x1, y1 - 70), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                        cv2.putText(violation_frame, label, (x1, y1 - 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                        
                                        # Отправлять в очередь нарушение
                                        self.violation_queue.put((stream_id, track_id, violation_frame, label, local_time))
                                    except Exception as e:
                                        logging.error(f"Ошибка при обработке нарушения для трека {track_id if 'track_id' in locals() else 'unknown'}: {e}")
                            except Exception as e:
                                logging.error(f"Ошибка при обработке нарушений для потока {stream_id}: {e}")
                    except Exception as e:
                        logging.error(f"Ошибка при обработке кадра для потока {stream_id}: {e}")
                        continue

            # Сохранить только уникальные нарушения в Excel
            query = """
            SELECT DISTINCT track_id, violation_type, MIN(timestamp) AS first_occurrence
            FROM violations
            GROUP BY track_id, violation_type
            ORDER BY first_occurrence
            """
            data = pd.read_sql_query(query, conn)
            excel_path = "unique_violations.xlsx"
            data.to_excel(excel_path, index=False)
            logging.info(f"✅ Данные сохранены в {excel_path}")
            

            zip_path = "violation_frames.zip"
            shutil.make_archive("violation_frames", 'zip', "violation_frame")
            logging.info(f"✅ Папка с фотографиями архивирована в {zip_path}")
            print(f"✅ Папка с фотографиями архивирована в {zip_path}")

            conn.close()
            return "✅ Обработка завершена, Все данные сохранены и готовы для скачивания", excel_path, zip_path

        except Exception as e:
            if 'conn' in locals():
                conn.close()
            logging.error(f"Ошибка при обработке: {e}")
            return f"Ошибка при обработке: {str(e)}", None, None