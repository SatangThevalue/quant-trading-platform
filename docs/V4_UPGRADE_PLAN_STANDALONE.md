# V4 Standalone MQL5 Upgrade Plan (News Filter & Smart Execution)

แผนการอัปเกรดนี้มุ่งเน้นการแก้ปัญหาระบบ Quant ให้ทนทานต่อสภาวะตลาดจริง (โดยเฉพาะทองคำและ GBP) โดยออกแบบให้ไฟล์ **MQL5 EA สามารถรับมือข่าวเศรษฐกิจและสภาวะตลาดได้ด้วยตัวเอง 100% (Standalone)** โดยไม่ต้องพึ่งพาเซิร์ฟเวอร์ Python หลังบ้าน

---

## 🏗️ 1. MQL5 Architecture Upgrade (The "Smart EA" Plan)

ใน V4 EA จะไม่ได้มีแค่ Risk Engine ธรรมดา แต่จะถูกติดตั้ง "Radar" เพิ่มเติมเข้าไปในโค้ด `QuantTrader_V4.mq5`:

### 📡 โมดูล 1: Automated News Filter (ดักข่าวจาก ForexFactory)
- **กลไกการดึงข้อมูล:** EA จะยิงคำสั่ง `WebRequest()` ไปดึงไฟล์ XML จาก ForexFactory (`https://nfs.faireconomy.media/ff_calendar_thisweek.xml`) **เพียงแค่วันละ 1 ครั้ง** เพื่อป้องกันการถูกแบน IP
- **การจัดการเวลา (Dynamic Timezone Sync):** EA จะคำนวณ `TimeCurrent() - TimeGMT()` เพื่อชดเชยเวลาของ Broker กับเวลาข่าวอัตโนมัติ หมดปัญหาการหลงเวลา
- **News Blackout Logic:** 
  - ก่อนข่าว High Impact 30 นาที ➡️ สั่งปิด Inference ห้ามส่งออเดอร์ใหม่
  - หากมีออเดอร์เก่าค้างอยู่ ➡️ ปรับ Stop Loss เลื่อนมาบังทุน (Break-even) ทันที
  - หลังข่าวออก 15 นาที ➡️ EA ตื่นขึ้นมาทำงานต่อตามปกติ

### 🧠 โมดูล 2: The "Meta-Filter" (ตัวแทน Meta-Labeling ฝั่ง MQL5)
แทนที่จะโหลดโมเดล ONNX 2 ตัว (ซึ่งกิน RAM ของ MT5 มากเกินไป) เราจะเขียนโค้ด "การกรองแบบ Meta" ฝังใน EA ไปเลย:
- **Spread Checker ก่อนเข้าทำ:** EA จะคำนวณว่า "เป้าหมายกำไร (Take Profit) ห่างจากจุดเข้า กี่เท่าของค่า Spread ปัจจุบัน?" ถ้าน้อยกว่า 3 เท่า EA จะตีความว่าเป็นสัญญาณขยะ (Noise) และปฏิเสธออเดอร์นั้น (ทำงานเหมือน Meta-Model บนกราฟ 15m)

### 🛡️ โมดูล 3: Asymmetric Trade Management (สำหรับการเทรด Crypto & Gold)
- ลอจิก SL/TP แบบไม่สมมาตร: ถ้า EA รู้ว่ากำลังเทรด `BTCUSD` มันจะสับสวิตช์ไปใช้ `SL = 2.0 ATR` และ `TP = 5.0 ATR` อัตโนมัติ (ยอมตัดขาดทุนกว้างขึ้น แต่กวาดกำไรคำใหญ่ตามพฤติกรรมคริปโต)
- **Trailing Stop แบบบันได:** เมื่อกำไรวิ่งไปถึง `2.0 ATR` EA จะขยับ SL ไปล็อกกำไรไว้ที่ `1.0 ATR` และไล่ตามขึ้นไปเรื่อยๆ

---

## 🐍 2. Python Pipeline Upgrade (The V4 Focus)

เมื่อ MQL5 เก่งขึ้นฝั่ง Execution ฝั่ง Python จะมุ่งเน้นสกัดฟีเจอร์สำหรับสินทรัพย์จำเพาะ (Asset-Specific) ให้แม่นขึ้น:

1. **Fractional Differencing:** สำหรับทองคำ (GOLD) โค้ดสร้าง Feature จะถูกเปลี่ยนจากการใช้ RSI ปกติ เป็นการคำนวณ `FracDiff` ควบคู่กับค่าความผันผวนของแท่งเทียน เพื่อให้โมเดลจำเทรนด์ระยะยาวได้
2. **Hyperparameter Bound:** สำหรับกราฟเล็ก (15m/30m) Optuna จะถูกจำกัด `max_depth` ไม่เกิน 3 เพื่อป้องกันการจำ Noise

---

## 📈 3. Projected Performance (การคาดการณ์ประสิทธิภาพหลังอัปเกรด V4)

| Asset (15m) | V3 (No Hard Filter) | V4 Expected (News Filter + Smart Sizing) |
|-------------|---------------------|------------------------------------------|
| **EURUSD**  | Net: -11.5%         | **Net: +5% ถึง +15%** (รอด Spread เพราะ Limit Order) |
| **GOLD**    | DD: -76% (ล้างพอร์ต) | **DD: -15%** (รอดเพราะ News Blackout ก่อน NFP) |
| **BTCUSD**  | WinRate: 38%        | **WinRate: ~45%** (ทนสวิงล่า SL ได้ด้วย Asymmetric Risk) |

---

## 🛠️ 4. แผนการดำเนินงาน (Execution Steps)

1. **Step 1:** จัดทำไฟล์ `src/mql5/Include/NewsFilter.mqh` เพื่อเขียนคลาสจัดการการดึงและแปลงเวลาข่าว
2. **Step 2:** อัปเกรดร่าง EA ตัวเต็ม `src/mql5/QuantTrader_ONNX_V4.mq5` เพื่อประกอบร่าง News Filter, Trailing Stop และ Limit Order Logic เข้าไป
3. **Step 3:** ปรับปรุงเอกสารคู่มือ `PAPER_TRADING_CHECKLIST.md` เพื่อให้สอดคล้องกับการตั้งค่า URL ใน MT5

*(เอกสารแผนการและกรอบความคิดฉบับสมบูรณ์ พร้อมให้พิจารณาก่อนลงมือเขียนโค้ด)*