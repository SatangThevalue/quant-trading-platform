# Paper Trading Readiness Checklist

เอกสารสำหรับ PM และ Quant Researcher ใช้ประเมินความพร้อมของระบบ "AI Quant Trading Platform" ก่อนอนุญาตให้นำโมเดลไปผูกกับบัญชีทดลอง (Paper Trading / Demo Account) เพื่อรันสดในตลาดจริงเป็นเวลา 30 วัน

---

## 🟢 1. Data & Environment Readiness (ความพร้อมของข้อมูลและระบบ)
- [ ] **Data Pipeline อัตโนมัติทำงานได้จริง:** สคริปต์ `data_collection_flow.py` ดึงราคาปิดและ Spread ล่าสุดเข้ามาได้โดยไม่ Error
- [ ] **VPS / Local Server เสถียร:** มีอินเทอร์เน็ตที่เชื่อมต่อกับ Broker ตลอด 24 ชม. และสเปกเครื่องรองรับการทำ Inference
- [ ] **MT5 Terminal ล็อกอินสำเร็จ:** ล็อกอินเข้าบัญชี Demo ได้, ตั้งค่ายอมรับการรัน Algo Trading / DLL เรียบร้อยแล้ว

## 🟡 2. Model & Feature Readiness (ความพร้อมของโมเดล)
- [ ] **ONNX Model ถูก Export อย่างสมบูรณ์:** มีไฟล์ `model.onnx` จาก LightGBM พร้อมใช้
- [ ] **Feature Matching 100%:** รายชื่อฟีเจอร์ใน `feature_order.json` บน Python (เช่น `ret_5`, `rsi14`, `atr_ratio`) ถูกโค้ดไว้ใน `QuantTrader_ONNX.mq5` ใน **ลำดับเดียวกันเป๊ะ** และคำนวณด้วยสูตรเดียวกัน
- [ ] **Walk Forward Passed:** โมเดลทำผลงานย้อนหลัง (Out-of-Sample) ได้ Sharpe > 1.0 และ PF > 1.2 เป็นอย่างน้อย
- [ ] **Baseline Comparison:** มั่นใจแล้วว่า AI ทำกำไรได้เสถียรกว่าแค่การ "Buy & Hold"

## 🟠 3. Execution & Risk Readiness (ความพร้อมฝั่ง MQL5 EA)
- [ ] **Signal Threshold ตั้งค่าเหมาะสม:** ไม่ใช่ให้เทรดทุกครั้งที่ Prob > 0.5 แนะนำใช้ Buy > 0.60, Sell < 0.40
- [ ] **Position Sizing Engine ทำงานได้:** เช็คแล้วว่าระบบคำนวณ Lot จากเปอร์เซ็นต์ความเสี่ยงของพอร์ต (Risk %) และ ATR ได้ ไม่ได้ Fix ค่า `0.01` ตลอดเวลา
- [ ] **Spread Filter ทำงาน:** ป้องกันการยิงออเดอร์ในจังหวะตลาดสวิง/สภาพคล่องแห้ง (เช่น สเปรดพุ่งเกิน 30 จุด ระบบต้อง Reject)
- [ ] **SL / TP ครบถ้วน:** ทุกครั้งที่เปิดไม้ EA จะตั้ง Stop Loss (เช่น 1.5 * ATR) และ Take Profit (3.0 * ATR) ไปยัง Broker ทันที เผื่อกรณีเน็ตหลุด
- [ ] **Portfolio Rules (ถ้าเทรดหลายคู่):** ตรวจสอบว่ามีระบบป้องกันไม่ให้เปิดออเดอร์ USD ทางเดียวกันพร้อมกันหลายคู่

## 🔴 4. Monitoring Readiness (ความพร้อมระหว่างเทรดจำลอง)
- [ ] **MLflow (ถ้ามี):** สามารถเก็บประวัติการทายผลได้ หรืออย่างน้อย MT5 Journal (Experts Log) พิมพ์ค่า Probability และ Features ออกมาทุกครั้งที่เทรด
- [ ] **Tracking Dashboard:** เชื่อมบัญชี Demo เข้ากับ MyFxBook, MQL5 Signals, หรือ QuantStats เพื่อติดตาม Drawdown รายวัน

---

### ⏱️ กระบวนการ Paper Trading (Forward Testing)
เมื่อเช็คลิสต์ทั้งหมดผ่านแล้ว ให้ดำเนินการดังนี้:
1. **รันระบบ 24/5:** เปิดเครื่องให้ EA ทำงานแบบไม่ต้องจับตาดู (Hands-off) เป็นเวลา **อย่างน้อย 4 สัปดาห์ (30 วัน)**
2. **ห้ามแทรกแซง:** ไม่ว่าจะกำไรหรือขาดทุน ห้ามปิดออเดอร์ด้วยมือ หรือเข้าไปแก้โค้ดโมเดลกลางคัน (เพื่อเก็บ Data แท้จริง)
3. **การประเมินผล:** จบ 30 วัน นำ Log การเทรดจาก MT5 มาคำนวณ:
   - Profit Factor ในตลาดสดใกล้เคียงกับ Backtest หรือไม่?
   - Slippage ที่เจอจริงๆ กินกำไรไปกี่เปอร์เซ็นต์?
   - มีจังหวะไหนที่โมเดลส่งสัญญาณเพี้ยน (Concept Drift) หรือไม่?
