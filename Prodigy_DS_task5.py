import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib
matplotlib.use('TkAgg')  # Ensures plots show in VS Code

# Load dataset
df = pd.read_csv('US_Accidents_March23.csv')

# Basic info
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

# Drop unnecessary columns
df = df.drop(columns=['ID', 'Source', 'Description', 'Number', 'Street', 'City'])

# Convert Start_Time to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Extract date & time features
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.day_name()
df['Month'] = df['Start_Time'].dt.month

# Accidents by Hour
plt.figure(figsize=(12,6))
sns.countplot(x='Hour', data=df, palette='plasma')
plt.title('Accidents by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.tight_layout()
plt.show()

# Accidents by Day of Week
plt.figure(figsize=(12,6))
sns.countplot(x='Weekday', data=df, 
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
              palette='viridis')
plt.title('Accidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.tight_layout()
plt.show()

# Top Weather Conditions
plt.figure(figsize=(15,6))
top_weather = df['Weather_Condition'].value_counts().nlargest(10)
sns.barplot(x=top_weather.index, y=top_weather.values, palette='coolwarm')
plt.title('Top Weather Conditions during Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Accidents vs Visibility
plt.figure(figsize=(10,5))
sns.histplot(df['Visibility(mi)'], kde=True, color='orange')
plt.title('Accidents vs Visibility')
plt.tight_layout()
plt.show()

# Accidents vs Temperature
plt.figure(figsize=(10,5))
sns.histplot(df['Temperature(F)'], kde=True, color='skyblue')
plt.title('Accidents vs Temperature')
plt.tight_layout()
plt.show()

# HeatMap
df_map = df[['Start_Lat', 'Start_Lng']].dropna().sample(10000)
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
HeatMap(data=df_map.values, radius=10).add_to(m)
m.save("us_accident_hotspots.html")
print("Heatmap saved to us_accident_hotspots.html")

# Correlation heatmap
cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
df_corr = df[cols].dropna()

plt.figure(figsize=(10,6))
sns.heatmap(df_corr.corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Between Severity and Environmental Factors')
plt.tight_layout()
plt.show()
