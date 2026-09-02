# Deployment & Execution Checklist (EURUSD & USDCHF)

เอกสารนี้ใช้สำหรับตรวจสอบความพร้อมก่อนนำโมเดล `EURUSD` และ `USDCHF` (ซึ่งสอบผ่านการทำ Backtest ระดับ Grade A/B) ขึ้นสู่กระบวนการ Paper Trading บนโปรแกรม MetaTrader 5 โดยใช้ `QuantTrader_ONNX_V2.mq5`

---

## 🟢 1. File Transfer Readiness (การโอนย้ายไฟล์โมเดล)
ตรวจสอบว่ามีการคัดลอกไฟล์จากโฟลเดอร์ `models/production/` ไปยังโฟลเดอร์ `MQL5/Files/` บนเครื่องรัน MT5 ครบถ้วนหรือไม่:

### สำหรับ EURUSD
- [ ] `EURUSD_model_v1.0.onnx`
- [ ] `EURUSD_feature_order_v1.0.json`
- [ ] `EURUSD_model_config_v1.0.json`

### สำหรับ USDCHF
- [ ] `USDCHF_model_v1.0.onnx`
- [ ] `USDCHF_feature_order_v1.0.json`
- [ ] `USDCHF_model_config_v1.0.json`

---

## 🟡 2. MQL5 Environment Setup (การเตรียม MT5)
- [ ] ดาวน์โหลดและติดตั้งไลบรารี `JAson.mqh` ลงใน `MQL5/Include/` เรียบร้อยแล้ว
- [ ] คัดลอก `QuantTrader_ONNX_V2.mq5` ไปไว้ที่ `MQL5/Experts/`
- [ ] กด Compile `QuantTrader_ONNX_V2.mq5` แล้วต้องไม่มี Error แจ้งเตือน (Warning ปกติยอมรับได้)
- [ ] ใน MT5: ไปที่ Tools -> Options -> Expert Advisors -> **Allow algorithmic trading** (เปิดใช้งาน)
- [ ] ใน MT5: อนุญาตให้ EA โหลดไฟล์ DLL (Allow DLL imports)

---

## 🟠 3. Dynamic Configuration Validation (ตรวจสอบการโหลด JSON ของ EA)
เปิดกราฟ `EURUSD` และ `USDCHF` (แยกกราฟกัน) ลาก EA ไปใส่ และดูที่แถบ **"Experts" Log** ว่าขึ้นข้อความสำเร็จดังนี้หรือไม่:

- [ ] `Loaded Feature Order | Detected 38 Features.` (ต้องแสดงตัวเลข 38 ตามที่ถูกเทรนมา)
- [ ] `Loaded Config | Buy: 0.75 | Sell: 0.25 | SL: 1.5x | TP: 3.0x` (ต้องแสดงค่าที่อ่านจาก JSON ได้ถูกต้อง)
- [ ] `ONNX System Ready. Model requires 38 features.`

*⚠️ หากแสดงผลว่า "Failed to parse..." แสดงว่า EA หาไฟล์ JSON ไม่เจอ ให้เช็ค Path ในหน้า Inputs ของ EA อีกครั้ง*

---

## 🔴 4. Execution & Safety Validation (ด่านตรวจความปลอดภัยก่อนรันสด)
- [ ] บัญชีทดลอง (Demo) ที่ใช้เป็นประเภท **Raw Spread / ECN** (เพื่อผลลัพธ์การจำลอง Cost ที่แม่นยำ)
- [ ] ในหน้า Inputs: ยืนยันว่าตั้งค่า `InpMaxDailyLoss` ไม่เกิน 3.0% 
- [ ] ในหน้า Inputs: ตรวจสอบ `InpRiskPercent` ว่าตั้งค่าเป็น 1.0 (1%) ไม่มากเกินไป
- [ ] ทดลองปล่อยรันให้ผ่านพ้นช่วงที่มีการประกาศข่าว (News Filter) 1 ครั้ง เพื่อดูว่า EA งดเทรดตามที่สั่งหรือไม่

---

## 🚀 5. Performance Monitoring (หลังเปิด Paper Trading)
เมื่อรันแล้ว 1-2 สัปดาห์ ให้เก็บข้อมูลเพื่อนำมาเปรียบเทียบกับ Backtest Report ด้านล่างนี้:

### Benchmark (อ้างอิงจาก Out-of-Sample 1 ปีล่าสุด)
| Metric | EURUSD (Expected) | USDCHF (Expected) |
|--------|-------------------|-------------------|
| **Profit Factor** | 1.34 | 1.56 |
| **Win Rate** | ~54% | ~57% |
| **Max Drawdown** | < -10% | < -10% |

- [ ] ผลการรันบน Demo (Paper Trading) มีความใกล้เคียงหรือล้อไปกับ Benchmark ด้านบน 
- [ ] อาการ Slippage บน Broker ของจริง ไม่ได้กินกำไรของระบบจนทำให้กลายเป็นขาดทุน

---
*จัดทำโดย MLOps & Quant Architect*