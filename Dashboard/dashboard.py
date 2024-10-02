import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load /workspaces/dicoding-e-commerce/Data/order_reviews_dataset.csv
def load_data():
    orders_df = pd.read_csv('../Data/orders_dataset.csv')
    order_reviews_df = pd.read_csv('../Data/order_reviews_dataset.csv')
    
    return orders_df, order_reviews_df

# Processing
def clean_data(orders_df, order_reviews_df):
    success_orders_df = orders_df[orders_df['order_status'] == 'delivered'].dropna()
    order_reviews_df['review_comment_title'] = order_reviews_df['review_comment_title'].fillna('No Title')
    order_reviews_df['review_comment_message'] = order_reviews_df['review_comment_message'].fillna('No Message')

    # Convert date columns to datetime
    date_columns = [
        'order_delivered_customer_date',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_purchase_timestamp',
        'order_estimated_delivery_date'
    ]
    for col in date_columns:
        success_orders_df[col] = pd.to_datetime(success_orders_df[col], errors='coerce')

    # Create status column
    success_orders_df['status'] = success_orders_df.apply(
        lambda row: 'Early' if row['order_delivered_customer_date'].date() < row['order_estimated_delivery_date'].date() else
                     'Late' if row['order_delivered_customer_date'].date() > row['order_estimated_delivery_date'].date() else
                     'On Time', axis=1
    )
    success_orders_df['delivery_time_days'] = (success_orders_df['order_delivered_customer_date'] - success_orders_df['order_purchase_timestamp']).dt.days

    # Merge dataframes
    review_df = pd.merge(success_orders_df, order_reviews_df, how="left", on="order_id")
    
    return success_orders_df, review_df

# Visualization
def plot_status_counts(success_orders_df):
    status_counts = success_orders_df['status'].value_counts()[['Early', 'On Time', 'Late']]
    plt.figure(figsize=(10, 6))
    plt.bar(status_counts.index, status_counts.values, color=['#0d62ff', '#5792ff', '#8ab3ff'])
    plt.title('Status Pesanan', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, value in enumerate(status_counts.values):
        plt.text(i, value, value, ha='center', va='bottom')
    st.pyplot(plt)

def plot_delivery_time_distribution(success_orders_df):
    plt.figure(figsize=(10, 6))
    plt.hist(success_orders_df['delivery_time_days'], bins=30, color='#5792ff', alpha=0.6, edgecolor='black')
    plt.axvline(success_orders_df['delivery_time_days'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(success_orders_df['delivery_time_days'].median(), color='green', linestyle='dashed', linewidth=1, label='Median')
    plt.xlabel('Waktu Pengiriman (Hari)', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.title('Distribusi Waktu Pengiriman', fontsize=16)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

def plot_delivery_time_vs_review(review_df):
    average_delivery_time = review_df.groupby('review_score')['delivery_time_days'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=average_delivery_time, x='review_score', y='delivery_time_days', color='#5792ff')
    plt.title('Rata-Rata Waktu Pengiriman dengan Ulasan', fontsize=14)
    plt.ylabel('Rata-Rata (Hari)')
    plt.xlabel('Skor Ulasan')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Main Function
def main():
    st.title("Analisa Pengiriman dan Kepuasan Pelanggan ✨")
    st.caption("| Hauzan Ariq Bakri | Oktober 2024 |")
    st.write("")
    orders_df, order_reviews_df = load_data()
    success_orders_df, review_df = clean_data(orders_df, order_reviews_df)

    # Dashboard Sections
    if st.checkbox('Data Pesanan'):
        st.write(orders_df.sample(5))

    if st.checkbox('Data Ulasan'):
        st.write(order_reviews_df.sample(5))

    # 1
    st.subheader("Status Pesanan")
    st.markdown(
        """
        Banyak pesanan yang sampai lebih awal dari perkiraan estimasi perjalanan dengan nilai **88630**, dilanjutkan dengan pesanan telat yang berjumlah cukup banyak yaitu **6533**, dan pesanan yang sampai tepat waktu sebanyak **1292**.
        """
    )
    st.write(success_orders_df.groupby(by="status").order_id.nunique().sort_values(ascending=False))
    plot_status_counts(success_orders_df)

    # 2
    st.subheader("Waktu Pengiriman")
    st.markdown(
        """
        Selanjutnya kita akan melihat berapa lama sebuah pesanan sampai ke tujuan (dari pelanggan membayar pesanan), kita akan menghitung selisih dari *order_delivered_customer_date* (tanggal sampai) dan *order_purchase_timestamp* (tanggal pembelian), nantinya hasil akan dimasukkan ke kolom *delivery_time_day*
        """
    )
    st.write(success_orders_df['delivery_time_days'].describe())
    st.markdown(
        """
        Dari rangkuman statistik tersebut kita dapat melihat beberapa poin penting, dari 96455 data, pesanan memiliki rata-rata waktu **12 hari** perjalanan dengan pesanan yang paling cepat adalah **dihari yang sama (0 hari)** dan pesanan yang paling lama sampai bernilai **209 hari**.
        """
    )
    plot_delivery_time_distribution(success_orders_df)

    # 3
    st.subheader("Hubungan Pengiriman dengan Kepuasan Pelanggan")
    st.markdown(
        """
        Melakukan analisa untuk melihat hubungan atnara kecepatan pengiriman dengan skor ulasan, pertama-tama kita akan lakukan pengecekkan korelasi antara kolom *delivery_time_days* dengan *review_score*.
        """
    )
    
    correlation = review_df[['delivery_time_days', 'review_score']].corr()
    st.write(correlation)
    st.markdown(
        """
        Ternyata lamanya pengiriman pesanan ***Berkolerasi Negatif Sedang*** dengan skor ulasan, dengan begitu menunjukkan bahwasannya ketika nilai  *delivery_time_days* naik, maka nilai *review_score* cenderung menurun, begitu pula sebaliknya.
        """
    )
    
    st.markdown(
        """
        Kita dapat membuat kategori rentang (***bins***) hari perjalanan pesanan dengan menggunakan fungsi ***cut()*** milik pandas, kita akan mendefinisikan rentang sebanyak 5 atau 10 interval nantinya data kategori akan dimasukkan ke kolom *delivery_time_category*. Kemudian kategori tersebut akan di tampilkan menggunakan pivot table beserta dengan rata-rata skor ulasan.
        """
    )
    review_df['delivery_time_category'] = pd.cut(
        review_df['delivery_time_days'],
        bins=[0, 5, 10, 15, 20, 30, 40, 100],
        labels=['0-5', '6-10', '11-15', '16-20', '21-30', '31-40', '40+']
    )
    st.write(review_df.groupby('delivery_time_category').agg({'review_score': 'mean', 'order_id': 'count'}).reset_index())
    st.markdown(
        """
        Penurunan skor ulasan tidak dapat dipungkiri lagi, bisa dilihat penurunan yang signifikan terjadi antara rata-rata skor sepanjang kategori hari lamanya proses pengiriman.
        """
    )
    
    plot_delivery_time_vs_review(review_df)

    # 4
    st.subheader("Kesimpulan")
    st.markdown(
        """
        - Apakah pesanan sampai ke tujuan sesuai dengan prediksi atau sebaliknya? Pesanan rata-rata **tiba lebih awal** dibandingkan dengan tanggal prediksi.
        - Seberapa lama pesanan sampai ke tangan pelanggan? Rata-rata waktu pengiriman adalah 12 hari.
        - Adakah hubungan kecepatan pengiriman dengan kepuasan pelanggan? Ada hubungan negatif sedang antara waktu pengiriman dan skor ulasan.
        """
    )

if __name__ == "__main__":
    main()
