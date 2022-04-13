
<center><h1>Logistic Regression</center>
    
## Giới thiệu thuật toán Logistic Regression:

là một thuật toán phân loại.Logistic Regression là mô hình hồi quy nhằm dự đoán giá trị đầu ra rời rạc (discrete target variable) y tương ứng với một vecto đầu vào X. Việc này tương đương với chuyện phân loại các đầu vào X vào các nhóm y tương ứng. Nói một cách dễ hiểu Logistic Regression là một kỹ thuật phân loại một tập dữ liệu dựa trên các giá trị đầu vào.

Giả sử chúng ta có bộ dữ liệu phân tích để hiểu khách hàng nào sẽ rời đi vào tháng tới:

<p align="center">
    <img src="https://i.imgur.com/wYoaEYe.png">
</p>
Giả sử bạn là một nhà phân tích của công ty này và bạn phải tìm ra ai đang rời đi và tại sao?
Bạn sẽ sử dụng tập dữ liệu để xây dựng mô hình dựa trên ghi chép lịch sử và sử dụng nó để dự đoán xu hướng trong tương lai của nhóm khách hàng.
     Bộ dữ liệu sẽ bao gồm thông tin về các dịch vụ mỗi khách hàng đã đăng ký, thông tin tài khoản khách hàng, thông tin nhân khẩu học như giới tính độ tuổi và cả những thông tin của khách hàng đã rời đi hoặc không trong tháng vừa qua (cột churn) .
 Trong Logistic Regression, chúng ta sử dụng một hoặc nhiều biến độc lập như “tenure” , “age”, “income” … để dự đoán một kết quả, chẳng hạn như “churn” mà chúng ta gọi là biến phụ thuộc đại diện cho việc liệu khách hàng sẽ ngừng sử dụng dịch vụ hay không.
    
## Ứng dụng của Logistic regression

Vì nó là một loại thuật toán phân loại, vì vậy nên nó có thể sử dụng cho các tình huống khác nhau:

* Dự đoán xác suất của một người lên cơn đau tim theo thời gian xác định dựa trên thông tin của chúng ta về tuổi của người đó, giới tính, cân nặng của cơ thể.
<p align="center">
    <img src="https://i.imgur.com/VqEoEyG.png"/>

</p>
    
* Dự đoán khả năng bị bệnh tiểu đường dựa trên đặc điểm quan sát như cân nặng, chiều cao, huyết áp, kết quả xét nghiệm máu…
* Hoặc trong marketing, chúng ta có thể sử dụng nó để dự đoán khả  năng mua hàng của khách hàng mua một sản phẩm.
<p align="center">
    <img src="https://i.imgur.com/h6bDsIG.png"/>
</p>
    
* Chúng ta cũng có thể dùng hồi quy logistic để dự đoán xác suất thất bại của một quy trình, hệ thống hoặc sản phẩm nhất định. Thậm chí có thể dùng nó để dự đoán khả năng chủ nhà vỡ nợ trên một thế chấp…

<p align="center">
    <img src="https://i.imgur.com/3Auwm8f.png"/>
</p>
    
## Điểm giống và khác so với Linear Regression
Cả hai phương pháp đề là môn hình học có giám sát.
Logistic Regression tương tự như Linear Regression nhưng nó cố dự đoán trường mục tiêu phân loại hay rời rạc . Trong Linear Regression chúng ta có thể cố gắng dự đoán một giá trị liên tục của các biến như: giá của một căn nhà, huyết áp của bệnh nhân, hoặc lượng tiêu thụ nhiên liệu của một chiếc xe… Nhưng trong Logistic Regression, chúng ta dự đoán một biến nhị phân như có/không, đúng/sai, thành công/không thành công, có thai/không có thai… Tất cả đều có thể mã hóa thành dạng 0, 1.
    
## Phương pháp

<!-- Ý tưởng của bài toán là gì? -->
Mục đích của các mô hình máy học à tìm ra con số $\theta$.Sao cho tổng độ lỗi có giá trị nhỏ nhất.

$$
    L(\theta) = argmin_\theta\sum_{i=1}^{n}D(f_\theta(x^{(i)}),y^{(i)})
$$
Nếu sử dụng côn thức của linear regression.
$$
    L(\theta) = argmin_\theta\sum_{i=1}^{n}D(f_\theta(x^{(1)}) - y^{(i)})
$$
Thì chúng ta sẽ gặp phải một số vấn đề về miền giá trị.y nằm trong khoảng [0,1].Giá trị của y~ thì nằm trong khoảng [$-\infty$,$+\infty$].Công thức đó sẽ bị phụ thuộc vào y~ hơn là bị phụ thuộc vào y.Tập trung tìm $\theta$ sao cho y~ nhỏ nhất thay vì tìm $\theta$ sao cho y~ xấp xỉ bằng y.

Ý tưởng của Logistics Regression là sử dụng một hàm sigmoid :
$$f(x) =  \frac{1}{\ 1 + \mathrm{e}^{-x}}$$

Lý do mà hàm sigmoid được chọn bởi vì hàm sigmoid là hàm số liên tục nhận giá trị thực.Bị chặn trong khoảng [0,1].Cho dù là tiến thằng đến $-\infty$ hay là $+\infty$. Thì giá trị của hàm sigmoid đề bị giới hạn trong khoảng [0,1].

<p algin="center">
<img src="https://i.imgur.com/nwyNC2q.png" />
</p>

Công thức hàm độ lỗi của hàm sigmoid sẽ có dạng :
$$
   L(\theta) = argmin_\theta-\sum_{i=1}^{n}y^{(i)}logf_\theta(x^{(i)}) + (1-y^{(i)})log(1-f_\theta(x^{(i)}))
$$


Logistic Regression được sử dụng nhiều trong bài toán phân lớp nhị phân.Dữ liệu Logistic Regression tuyến tính.

Dữ liệu đầu vào cho mô hình Logistic Regression thường là các bức ảnh (ma trận số), số liệu các thuộc tính của từng điểm dữ liệu.
Các điểm dữ liệu đôi khi được đo đạc với những đơn vị khác nhau hoặc có hai thành phần của vecto dữ liệu chênh lệch nhau quá lớn, một thành phần có khoảng giá trị từ 0 tới 1000, thành phần kia chỉ có khoảng giá trị từ 0 đến 1. Đôi khi, chúng ta cần chuẩn hóa dữ liệu trước khi thực hiện các bước tiếp theo

### Ưu và nhược điểm của Logistic Regression.

<p align="center">
    
| Ưu điểm  | Nhược điểm  |
|---|---|
| Logistic regression dễ thực hiện, dễ giải thích và rất hiệu quả để training.  |  Nếu số lượng quan sát ít hơn số lượng tính năng, Hồi quy Hậu cần không nên được sử dụng, nếu không, nó có thể dẫn đến overfitting. |
| Nó không đưa ra giả định về phân phối các lớp trong không gian tính năng.  |  Nó xây dựng các ranh giới tuyến tính. |
|  Nó có thể dễ dàng mở rộng đến nhiều lớp (multinomial regression) và một cái nhìn xác suất tự nhiên về dự đoán lớp.     |  Hạn chế chính của Hồi quy Logistic là giả định về tuyến tính giữa biến phụ thuộc và các biến độc lập. |
|  Nó không chỉ cung cấp một thước đo mức độ thích hợp của một dự đoán (kích thước hệ số) là, mà còn là hướng liên kết của nó (tích cực hoặc tiêu cực). |  Nó chỉ có thể được sử dụng để dự đoán các chức năng rời rạc. Do đó, biến phụ thuộc của Hồi quy Hậu cần bị ràng buộc với tập số rời rạc. |
|Nó rất nhanh trong việc phân loại các hồ sơ không xác định. | Các vấn đề phi tuyến tính không thể được giải quyết bằng hồi quy hậu cần vì nó có bề mặt quyết định tuyến tính. Dữ liệu có thể tách rời tuyến tính hiếm khi được tìm thấy trong các tình huống thực tế.  |
|Độ chính xác tốt cho nhiều tập dữ liệu đơn giản và nó hoạt động tốt khi bộ dữ liệu có thể tách rời theo tuyến tính.|Hồi quy hậu cần đòi hỏi sự đa đối tác trung bình hoặc không có sự đa đối tác giữa các biến độc lập.  |
| Nó có thể giải thích các hệ số mô hình là các chỉ số về tầm quan trọng của tính năng.  |  Thật khó để có được các mối quan hệ phức tạp bằng cách sử dụng hồi quy hậu cần. Các thuật toán mạnh mẽ và nhỏ gọn hơn như Neural Networks có thể dễ dàng vượt trội hơn thuật toán này. |



</p>

## Cài đặt chương trình
```infor
Tất cả các code phải có chú thích đầy đủ.
```
Để hiểu rõ hơn về mô hình Losgistic Regression.Thì chúng ra sẽ tiến hành viết mã để có thể hiểu cách triển khải nó.Ngôn ngữ được sử dụng ở đây là python.
Đầu tiên chúng ta sẽ thêm các thư viện cần thiết.
```python
    import numpy as np 
    from numpy import log,dot,e,shape
    import matplotlib.pyplot as plt
```
Chúng ta sẽ sử dụng thư viện sklearn để tạo dataset.
```python
    from sklearn.datasets import make_classification
    # Tạo dataset với 4 đặc trưng và 2 lớp
    X,y = make_classification(n_features = 4,n_classes=2)
```
Tiến hành chia tập dataset.
```python
    from sklearn.model_selection import train_test_split  
    # Chia tập train và test với tỷ lệ train 0.9 và test 0.1
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)
```
Hàm sigmoid
```python
    def sigmoid(self,z):
        sig = 1/(1+e**(-z))
        return sig
```
Hàm độ lỗi dữa vào lý thuyết ở trên
```python
      def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y) 
            return cost
```

Tiến hành áp dụng thuật toán Gradient Descent
```python
    # tạo ma trận để lưu các giá trị cần thiêt
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    # tiến hành sử dụng gradient descent 
    def fit(self,X,y,alpha=0.001,iter=100):
        params,X = self.initialize(X)
        cost_list = np.zeros(iter,)
        for i in range(iter):
            tmp = np.reshape(y,(len(y),1))
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))- tmp)
            cost_list[i] = cost(params)
        self.params = params
        return cost_list
```
Cài đặt hàm dự đoán cho mô hình.

```python
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
```
Cài đặt hàm để tính F1-Score
```python
    def F1_score(y,y_hat):
        tp,tn,fp,fn = 0,0,0,0
        for i in range(len(y)):
            if y[i] == 1 and y_hat[i] == 1:
                tp += 1
            elif y[i] == 1 and y_hat[i] == 0:
                fn += 1
            elif y[i] == 0 and y_hat[i] == 1:
                fp += 1
            elif y[i] == 0 and y_hat[i] == 0:
                tn += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*precision*recall/(precision+recall)
        return f1_score
```
Tiến hành chạy và đánh giá mô hình
```python
    obj1 = LogidticRegression()
    model= obj1.fit(X_tr,y_tr)
    y_pred = obj1.predict(X_te)
    y_train = obj1.predict(X_tr)
    #Let's see the f1-score for training and testing data
    f1_score_tr = F1_score(y_tr,y_train)
    f1_score_te = F1_score(y_te,y_pred)
    print(f1_score_tr)
    print(f1_score_te)
```
kết quả thu được sau khi chạy mô hình:
* f1_score tập train : 0.9777777777777777
* f1_score tập test : 1.0
<p align="center">
    <img src = "https://i.imgur.com/fcEYgYV.png" />
</p>

=> Toàn bộ code : [code](https://colab.research.google.com/drive/1UMYFOMk0Bnv_ai5gTy8ThqNGVkbbqeaH?usp=sharing)

## Bài tập
Xây dựng mô hình Logistic Regression cho bài toán phân lớp trên bộ dữ liệu dataset.csv được đính kèm trong file csv bên dưới.
Các bước thực hiện:
* Bước 1: Import các thư viện và đọc file csv
* Bước 2: Trích xuất ra các đặc trưng quan trọng từ file csv để đưa vào mô hình
* Bước 3: Phân chia bộ dữ liệu thành 75% train và 25% test
* Bước 4: Scaler dữ liệu
* Bước 5: Xây dựng mô hình phân lớp Logistic Regresstion từ bộ dữ liệu đã tiền xử lý ở trên
* Bước 6: Dự đoán trên bộ dữ liệu test
* Bước 7: Kiểm tra hiệu suất của mô hình
* Bước 8: Trực quan hóa hiệu suất của mô hình phân lớp trên.

=> link dataset : [dataset](https://drive.google.com/drive/folders/19Kkjc2sluXKVy-g5oISlW1ldYHCAYj7A?usp=sharing)
## Tổng kết:
Logistic Regression là 1 thuật toán phân loại được dùng để gán các đối tượng cho 1 tập hợp giá trị rời rạc (như 0, 1, 2, ...). Thuật toán trên dùng hàm sigmoid logistic để đưa ra đánh giá theo xác suất. Ví dụ: Khối u này 80% là lành tính, giao dịch này 90% là gian lận,...
    
Khi nào nên sử dụng Logistic Regression:
* Thứ nhất, khi đầu ra của dữ liệu là phân loại hoặc cụ thể là nhị phân. Chẳng hạn như 0/1, YES/NO, True/False …
* Thứ hai, khi bạn cần xác suất cho dự đoán của bạn. Ví dụ như nếu bạn muốn biết xác suất của khách hàng mua sản phẩm là gì. Hồi quy Logistic trả về điểm xác suất từ 0 đến 1 cho một mẫu dữ liệu nhất định. Trong thực tế, hồi quy logistic dự đoán xác suất của mẫu đó và chúng ta ánh xạ các trường hợp đến một lớp rời rạc dựa trên xác suất đó.
* Thứ ba, nếu dữ liệu của bạn được phân tách tuyến tính. Ranh giới của hồi quy logistic là đường thẳng hoặc mặt phẳng, hoặc siêu mặt phẳng. Một trình phân loại sẽ phân loại tất cả các điểm vào một bên của ranh giới quyết định như là thuộc về một lớp, và tất cả những điểm ở bên kia sẽ thuộc về một lớp khác. 
* Thứ tư, bạn cần hiểu tác động của một thuộc tính. Bạn có thể chọn thuộc tính tốt nhất dựa trên ý nghĩa thống kê của các hệ số hoặc tham số của mô hình hồi quy logistic. 

## Thành viên trong nhóm

    
| STT | MSSV | Họ và Tên |
|---|---|---|
| 1 | 19521322 | Huỳnh Ngọc Công Danh |
| 2 | 19520305 | Cao Đức Trí |
| 3 | 19521858 | Võ Tuấn Minh |
