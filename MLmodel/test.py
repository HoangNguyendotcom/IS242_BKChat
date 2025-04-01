import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# 1. Đọc dữ liệu
def load_data(file_path, sample_size=None):
    """
    Đọc dữ liệu từ file CSV và lấy mẫu nếu cần
    """
    df = pd.read_csv(file_path)
    print(f"Đã đọc {len(df)} dòng dữ liệu.")
    
    # Kiểm tra và hiển thị thông tin về các cột nhãn
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print("\nPhân bố nhãn trong dữ liệu:")
    for col in label_columns:
        if col in df.columns:
            positive_count = df[col].sum()
            print(f"{col}: {positive_count} ({positive_count/len(df)*100:.2f}%)")
    
    if sample_size and sample_size < len(df):
        print(f"\nLấy mẫu {sample_size} dòng dữ liệu...")
        # Đảm bảo mẫu bao gồm đủ các trường hợp của mỗi nhãn
        # Đầu tiên lấy một số lượng nhỏ mẫu từ mỗi nhãn
        sampled_indices = set()
        min_samples_per_label = min(100, sample_size // (len(label_columns) * 2))
        
        for col in label_columns:
            if col in df.columns:
                positive_samples = df[df[col] == 1].sample(
                    min(min_samples_per_label, df[col].sum()),
                    random_state=42
                ).index
                sampled_indices.update(positive_samples)
        
        # Sau đó lấy ngẫu nhiên các mẫu còn lại
        remaining_samples = sample_size - len(sampled_indices)
        if remaining_samples > 0:
            remaining_indices = df.index.difference(sampled_indices)
            additional_samples = np.random.choice(
                remaining_indices,
                min(remaining_samples, len(remaining_indices)),
                replace=False
            )
            sampled_indices.update(additional_samples)
        
        df = df.loc[list(sampled_indices)]
        print(f"Đã lấy mẫu {len(df)} dòng dữ liệu.")
    
    return df

# 2. Tiền xử lý văn bản
def preprocess_text(text):
    """
    Tiền xử lý cơ bản cho văn bản:
    - Chuyển về chữ thường
    - Loại bỏ URL, tag người dùng
    - Xóa tất cả dấu câu và ký tự đặc biệt
    """
    if pd.isna(text):
        return ""
    
    # Chuyển sang chữ thường
    text = str(text).lower()
    
    # Loại bỏ URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ tag người dùng (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Xóa tất cả dấu câu và ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Xây dựng model đa nhãn
def build_multilabel_model(X_train, y_train):
    """
    Xây dựng pipeline model cho phân loại đa nhãn với TF-IDF và LogisticRegression
    """
    # Sử dụng TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams và bigrams
        max_features=50000,
        min_df=2,
        max_df=0.8
    )
    
    # Sử dụng LogisticRegression với MultiOutputClassifier
    classifier = MultiOutputClassifier(
        LogisticRegression(
            C=5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='liblinear'  # Tốc độ nhanh hơn cho dữ liệu nhỏ
        )
    )
    
    # Tạo pipeline
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Huấn luyện model
    print("Đang huấn luyện model đa nhãn...")
    model.fit(X_train, y_train)
    print("Hoàn thành huấn luyện model.")
    
    return model

# 4. Đánh giá model đa nhãn
def evaluate_multilabel_model(model, X_test, y_test, label_names):
    """
    Đánh giá hiệu suất của model đa nhãn
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá tổng thể
    print("\nTổng quan hiệu suất model:")
    accuracy = (y_pred == y_test).mean()
    print(f"Độ chính xác trung bình: {accuracy:.4f}")
    
    # Đánh giá chi tiết cho từng nhãn
    for i, label in enumerate(label_names):
        print(f"\nBáo cáo chi tiết cho nhãn '{label}':")
        print(classification_report(y_test[:, i], y_pred[:, i]))
        
        # Vẽ ma trận nhầm lẫn cho từng nhãn
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.xlabel('Dự đoán')
        plt.ylabel('Thực tế')
        plt.title(f'Ma trận nhầm lẫn cho nhãn {label}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{label}.png')
        plt.close()
    
    return y_pred

# 5. Trích xuất từ ngữ nhạy cảm cho từng nhãn
def extract_toxic_words_by_label(model, label_names):
    """
    Trích xuất các từ có trọng số cao nhất cho từng nhãn
    """
    vectorizer = model.named_steps['vectorizer']
    classifier = model.named_steps['classifier']
    
    # Lấy feature names từ vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Lưu trữ kết quả
    toxic_words_by_label = {}
    
    # Trích xuất từ ngữ nhạy cảm cho từng nhãn
    for i, label in enumerate(label_names):
        # Lấy coefficients cho nhãn dương tính (positive class)
        coefficients = classifier.estimators_[i].coef_[0]
        
        # Tạo DataFrame với từ và trọng số tương ứng
        word_coefficients = pd.DataFrame({
            'word': feature_names,
            'coefficient': coefficients
        })
        
        # Sắp xếp giảm dần theo trọng số
        word_coefficients = word_coefficients.sort_values('coefficient', ascending=False)
        
        # Lọc các từ có trọng số dương (liên quan đến nhãn dương tính)
        toxic_words = word_coefficients[word_coefficients['coefficient'] > 0]
        
        # Lưu vào từ điển
        toxic_words_by_label[label] = toxic_words
    
    return toxic_words_by_label

# 6. Phân tích comments theo từng nhãn
def analyze_comments_by_label(texts, y_pred, label_names, top_n=30):
    """
    Phân tích từ ngữ phổ biến trong comments cho từng nhãn
    """
    common_words_by_label = {}
    
    # Chuyển texts thành list để tránh lỗi khi truy cập index
    texts_list = texts.values if hasattr(texts, 'values') else list(texts)
    
    for i, label in enumerate(label_names):
        # Lấy các comments được dự đoán dương tính cho nhãn này
        positive_indices = np.where(y_pred[:, i] == 1)[0]
        positive_texts = [texts_list[idx] for idx in positive_indices]
        
        if len(positive_texts) == 0:
            print(f"Không có comments nào được dự đoán dương tính cho nhãn '{label}'")
            common_words_by_label[label] = []
            continue
        
        # Tách từ
        all_words = []
        for text in positive_texts:
            words = text.split()
            all_words.extend(words)
        
        # Đếm tần suất
        word_counts = Counter(all_words)
        
        # Loại bỏ stop words đơn giản
        stop_words = set(['là', 'và', 'của', 'có', 'không', 'trong', 'cho', 'một', 'với', 'để', 'các', 'được',
                          'the', 'to', 'a', 'in', 'of', 'and', 'is', 'that', 'for', 'on', 'it', 'with'])
        for word in stop_words:
            if word in word_counts:
                del word_counts[word]
        
        # Lấy top N từ phổ biến nhất
        common_toxic_words = word_counts.most_common(top_n)
        common_words_by_label[label] = common_toxic_words
    
    return common_words_by_label

# 7. Hiển thị và lưu kết quả
def visualize_toxic_words_by_label(toxic_words_by_label, top_n=20):
    """
    Tạo biểu đồ hiển thị từ ngữ nhạy cảm cho từng nhãn
    """
    for label, words_df in toxic_words_by_label.items():
        if len(words_df) == 0:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Lấy top N từ
        top_words = words_df.head(top_n)
        
        # Vẽ biểu đồ
        sns.barplot(x='coefficient', y='word', data=top_words)
        plt.title(f'Top {top_n} từ ngữ nhạy cảm cho nhãn {label}')
        plt.xlabel('Trọng số')
        plt.ylabel('Từ')
        plt.tight_layout()
        plt.savefig(f'toxic_words_{label}.png')
        plt.close()

def visualize_common_words_by_label(common_words_by_label, top_n=20):
    """
    Tạo biểu đồ hiển thị từ ngữ phổ biến cho từng nhãn
    """
    for label, words in common_words_by_label.items():
        if len(words) == 0:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Chuyển sang DataFrame để dễ vẽ
        df = pd.DataFrame(words[:top_n], columns=['word', 'count'])
        
        # Vẽ biểu đồ
        sns.barplot(x='count', y='word', data=df)
        plt.title(f'Top {top_n} từ ngữ phổ biến trong comments nhãn {label}')
        plt.xlabel('Số lần xuất hiện')
        plt.ylabel('Từ')
        plt.tight_layout()
        plt.savefig(f'common_words_{label}.png')
        plt.close()

def save_results(toxic_words_by_label, common_words_by_label):
    """
    Lưu kết quả vào file
    """
    # Tạo báo cáo tổng hợp
    for label in toxic_words_by_label:
        # Lưu danh sách từ ngữ nhạy cảm theo hệ số
        if len(toxic_words_by_label[label]) > 0:
            toxic_words_by_label[label].to_csv(f'toxic_words_{label}.csv', index=False)
        
        # Lưu danh sách từ ngữ phổ biến
        if len(common_words_by_label[label]) > 0:
            pd.DataFrame(common_words_by_label[label], columns=['word', 'count']).to_csv(
                f'common_words_{label}.csv', index=False
            )
        
        # Tạo báo cáo kết hợp
        if len(toxic_words_by_label[label]) > 0 and len(common_words_by_label[label]) > 0:
            max_rows = max(len(toxic_words_by_label[label].head(50)), len(common_words_by_label[label][:50]))
            
            report_data = {
                'Từ ngữ nhạy cảm': toxic_words_by_label[label]['word'].head(max_rows).tolist(),
                'Hệ số': toxic_words_by_label[label]['coefficient'].head(max_rows).tolist(),
            }
            
            if len(common_words_by_label[label]) > 0:
                common_words = [word for word, _ in common_words_by_label[label][:max_rows]]
                common_counts = [count for _, count in common_words_by_label[label][:max_rows]]
                
                # Đảm bảo độ dài bằng nhau bằng cách thêm giá trị rỗng
                if len(common_words) < max_rows:
                    common_words.extend([''] * (max_rows - len(common_words)))
                    common_counts.extend([0] * (max_rows - len(common_counts)))
                
                report_data['Từ ngữ phổ biến'] = common_words
                report_data['Số lần xuất hiện'] = common_counts
            
            report = pd.DataFrame(report_data)
            report.to_csv(f'toxic_report_{label}.csv', index=False)
    
    print("Đã lưu các báo cáo phân tích từ ngữ nhạy cảm.")

# 8. Hàm chính
def main(data_path, sample_size=None):
    # Đọc dữ liệu
    print("Đọc dữ liệu từ", data_path)
    df = load_data(data_path, sample_size)
    
    # Kiểm tra và lấy danh sách nhãn
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_labels = [col for col in label_columns if col in df.columns]
    
    if not available_labels:
        print("Không tìm thấy các cột nhãn. Vui lòng kiểm tra dữ liệu đầu vào.")
        return None, None
    
    print(f"Các nhãn sẽ được sử dụng: {available_labels}")
    
    # Tiền xử lý văn bản
    print("Tiền xử lý văn bản...")
    df['processed_text'] = df['translated_comment_text'].apply(preprocess_text)
    
    # Kiểm tra và in một số mẫu
    print("\nMột số mẫu dữ liệu sau khi tiền xử lý:")
    sample_data = df[['processed_text'] + available_labels].sample(min(5, len(df)))
    print(sample_data)
    
    # Chia tập dữ liệu
    print("\nChia tập dữ liệu...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df[available_labels].values,
        test_size=0.2, 
        random_state=42
    )
    
    # Reset index để đảm bảo index bắt đầu từ 0 và liên tục
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    print(f"Kích thước tập huấn luyện: {len(X_train)}")
    print(f"Kích thước tập kiểm thử: {len(X_test)}")
    
    # Xây dựng model
    model = build_multilabel_model(X_train, y_train)
    
    # Đánh giá model
    print("\nĐánh giá model...")
    y_pred = evaluate_multilabel_model(model, X_test, y_test, available_labels)
    
    # Trích xuất từ ngữ nhạy cảm
    print("\nTrích xuất từ ngữ nhạy cảm cho từng nhãn...")
    toxic_words_by_label = extract_toxic_words_by_label(model, available_labels)
    
    # Phân tích comments
    print("\nPhân tích từ ngữ phổ biến trong comments cho từng nhãn...")
    common_words_by_label = analyze_comments_by_label(X_test, y_pred, available_labels)
    
    # Hiển thị kết quả
    print("\nTạo biểu đồ hiển thị kết quả...")
    visualize_toxic_words_by_label(toxic_words_by_label)
    visualize_common_words_by_label(common_words_by_label)
    
    # Lưu kết quả
    print("\nLưu kết quả phân tích...")
    save_results(toxic_words_by_label, common_words_by_label)
    
    # Trả về model và kết quả
    print("\nHoàn thành!")
    return model, (toxic_words_by_label, common_words_by_label)

# Để sử dụng:
if __name__ == "__main__":
    # Thay đổi đường dẫn tới dữ liệu của bạn
    model, results = main("translated_toxic_comment.csv", sample_size=1000)
    
    # Lưu model để sử dụng sau này
    if model is not None:
        joblib.dump(model, 'multilabel_toxic_model.joblib')
        print("Đã lưu model vào file 'multilabel_toxic_model.joblib'")
    
    # Phần bổ sung: Test model với câu nhập từ bàn phím
    print("\n===== TEST MODEL VỚI CÂU NHẬP TỪ BÀN PHÍM =====")
    
    # Kiểm tra xem model có sẵn chưa, nếu chưa thì load từ file
    if 'model' not in locals() or model is None:
        try:
            print("Đang load model từ file...")
            model = joblib.load('multilabel_toxic_model.joblib')
        except:
            print("Không tìm thấy model. Vui lòng huấn luyện model trước khi test.")
            exit()
    
    # Lấy tên các nhãn
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_labels = [col for col in label_columns if col in model.named_steps['classifier'].classes_]
    
    # Chế độ tương tác cho phép người dùng nhập văn bản
    print("\nNhập 'exit' hoặc 'quit' để thoát.")
    
    while True:
        # Nhận input từ người dùng
        user_input = input("\nNhập câu cần kiểm tra: ")
        
        # Kiểm tra nếu người dùng muốn thoát
        if user_input.lower() in ['exit', 'quit', 'thoát']:
            print("Kết thúc chương trình.")
            break
        
        # Tiền xử lý câu nhập vào
        processed_input = preprocess_text(user_input)
        
        # Dự đoán
        prediction = model.predict([processed_input])[0]
        
        # Hiển thị kết quả
        print(f"\nCâu gốc: \"{user_input}\"")
        print(f"Câu sau xử lý: \"{processed_input}\"")
        
        # Kiểm tra xem có bất kỳ nhãn nào dương tính không
        has_positive = np.any(prediction == 1)
        
        # Hiển thị nhãn toxic/normal tổng hợp
        if has_positive:
            print("=> KẾT LUẬN: TOXIC")
            print("Chi tiết các nhãn:")
            for i, label in enumerate(available_labels):
                if prediction[i] == 1:
                    print(f"- {label.upper()}: Có")
        else:
            print("=> KẾT LUẬN: NORMAL")
    
    print("\nQuá trình phân tích đã hoàn tất!")