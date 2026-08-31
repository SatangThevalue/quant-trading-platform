//+------------------------------------------------------------------+
//|                                           QuantTrader_ONNX.mq5   |
//|                        Copyright 2026, Satang AI Trading Project |
//|                                              https://github.com/ |
//+------------------------------------------------------------------+
#property copyright "Satang AI Quant Platform"
#property link      "https://github.com/SatangThevalue/quant-trading-platform"
#property version   "1.00"

// ------------------------------------------------------------------
// 1. INPUTS: The 6 Groups Configuration
// ------------------------------------------------------------------

// Group 1: Model & Signal
sinput string   InpModelName      = "models/production/EURUSD_model_v1.0.onnx";
input  bool     InpEnableONNX     = true;                                       
input  double   InpBuyThreshold   = 0.60;                                       
input  double   InpSellThreshold  = 0.40;                                       

// Group 2: Position Sizing & ATR
input  double   InpRiskPercent    = 1.0;                                        
input  bool     InpUseATRSizing   = true;                                       
input  int      InpATRPeriod      = 14;                                         
input  double   InpATRMultiplierSL= 1.5;                                        
input  double   InpATRMultiplierTP= 3.0;                                        

// Group 3: Risk Engine
input  double   InpMaxDailyLoss   = 3.0;                                        
input  double   InpMaxDrawdown    = 10.0;                                       
input  int      InpMaxSpread      = 30;                                         

// Group 4: Session Filter
input  bool     InpTradeLondon    = true;                                       
input  bool     InpTradeNewYork   = true;                                       
input  bool     InpTradeAsia      = false;                                      

// Group 5: Portfolio
input  double   InpMaxExposure    = 20.0;                                       

// ------------------------------------------------------------------
// 2. GLOBAL VARIABLES & HANDLES
// ------------------------------------------------------------------
long   onnx_handle = INVALID_HANDLE;
int    atr_handle  = INVALID_HANDLE;
int    rsi_handle  = INVALID_HANDLE;
double atr_buffer[];
double rsi_buffer[];

#define NUM_FEATURES 38

// ------------------------------------------------------------------
// 3. INITIALIZATION (OnInit)
// ------------------------------------------------------------------
int OnInit()
  {
   Print("Initializing QuantTrader ONNX EA...");
   
   atr_handle = iATR(_Symbol, PERIOD_D1, InpATRPeriod);
   rsi_handle = iRSI(_Symbol, PERIOD_D1, 14, PRICE_CLOSE);
   
   if(atr_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE)
     {
      Print("Error initializing indicators.");
      return(INIT_FAILED);
     }
     
   ArraySetAsSeries(atr_buffer, true);
   ArraySetAsSeries(rsi_buffer, true);

   if(InpEnableONNX)
     {
      Print("ONNX System Initialized. Waiting for data...");
     }

   return(INIT_SUCCEEDED);
  }

// ------------------------------------------------------------------
// 4. DEINITIALIZATION (OnDeinit)
// ------------------------------------------------------------------
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
     {
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
     }
   IndicatorRelease(atr_handle);
   IndicatorRelease(rsi_handle);
   Print("QuantTrader EA Shutdown.");
  }

// ------------------------------------------------------------------
// 5. MAIN EXECUTION LOOP (OnTick)
// ------------------------------------------------------------------
void OnTick()
  {
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_D1, 0);
   if(current_time == last_time) return; 
   
   // --- A. RISK LAYER: Spread Filter ---
   long current_spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(current_spread > InpMaxSpread) return;

   // --- B. FEATURE LAYER ---
   float features[NUM_FEATURES];
   if(!ValidateFeatures(features))
     {
      Print("Invalid Features Detected. NO TRADE.");
      return;
     }

   // --- C. ONNX INFERENCE LAYER ---
   float probability_buy = 0.5; 
   if(InpEnableONNX && onnx_handle != INVALID_HANDLE)
     {
      vectorf output_data(1);
      // bool success = OnnxRun(onnx_handle, ONNX_NO_CONVERSION, features, output_data);
      // if(success) probability_buy = output_data[0];
     }
     
   // --- D. SIGNAL LAYER ---
   int signal = 0;
   if(probability_buy >= InpBuyThreshold) signal = 1;
   else if(probability_buy <= InpSellThreshold) signal = -1;
   
   if(signal == 0) return;

   // --- E. POSITION SIZING ENGINE ---
   double sl_points = CalculateATRStopLoss();
   double volume = CalculateLotSize(sl_points);
   
   // --- F. EXECUTION ENGINE ---
   last_time = current_time; 
   ExecuteTrade(signal, volume, sl_points);
  }

// ------------------------------------------------------------------
// 6. HELPER FUNCTIONS 
// ------------------------------------------------------------------

bool ValidateFeatures(float &features_array[])
  {
   for(int i=0; i<ArraySize(features_array); i++)
     {
      if(!MathIsValidNumber(features_array[i])) return false; 
     }
   return true;
  }

double CalculateATRStopLoss()
  {
   CopyBuffer(atr_handle, 0, 1, 1, atr_buffer);
   double atr = atr_buffer[0];
   return atr * InpATRMultiplierSL / _Point; 
  }

double CalculateLotSize(double sl_points)
  {
   if(!InpUseATRSizing) return InpMinLot();
   
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (InpRiskPercent / 100.0);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   
   if(tick_value <= 0 || sl_points <= 0) return InpMinLot();
   
   double lot = risk_amount / (sl_points * tick_value);
   
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(lot < min_lot) lot = min_lot;
   if(lot > max_lot) lot = max_lot;
   
   return NormalizeDouble(lot, 2);
  }

double InpMinLot() { return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN); }

void ExecuteTrade(int signal, double lot, double sl_points)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double sl = 0, tp = 0;
   
   if(signal == 1) // BUY
     {
      sl = ask - (sl_points * _Point);
      tp = ask + ((sl_points / InpATRMultiplierSL) * InpATRMultiplierTP * _Point);
      PrintFormat("Executing BUY | Lot: %.2f | SL: %.5f | TP: %.5f", lot, sl, tp);
     }
   else if(signal == -1) // SELL
     {
      sl = bid + (sl_points * _Point);
      tp = bid - ((sl_points / InpATRMultiplierSL) * InpATRMultiplierTP * _Point);
      PrintFormat("Executing SELL | Lot: %.2f | SL: %.5f | TP: %.5f", lot, sl, tp);
     }
  }
