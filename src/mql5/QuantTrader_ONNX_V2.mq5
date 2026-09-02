//+------------------------------------------------------------------+
//|                                        QuantTrader_ONNX_V2.mq5   |
//|                        Copyright 2026, Satang AI Trading Project |
//|                                              https://github.com/ |
//+------------------------------------------------------------------+
#property copyright "Satang AI Quant Platform"
#property link      "https://github.com/SatangThevalue/quant-trading-platform"
#property version   "2.00"

// Include JAson library for JSON parsing
// Download from: https://www.mql5.com/en/code/25317
#include <JAson.mqh> 

// ------------------------------------------------------------------
// 1. INPUTS: Only Paths and Overrides
// ------------------------------------------------------------------
sinput string   InpModelPath      = "models/production/EURUSD_model_v1.0.onnx";
sinput string   InpConfigPath     = "models/production/EURUSD_model_config_v1.0.json";
sinput string   InpFeaturePath    = "models/production/EURUSD_feature_order_v1.0.json";

input  bool     InpEnableONNX     = true;                                       
input  double   InpRiskPercent    = 1.0;  // Fallback Risk %

// ------------------------------------------------------------------
// 2. GLOBAL VARIABLES & HANDLES
// ------------------------------------------------------------------
long   onnx_handle = INVALID_HANDLE;

// Variables loaded dynamically from JSON
int    num_features = 0;
double buy_threshold = 0.60;
double sell_threshold = 0.40;
double atr_sl_multiplier = 1.5;
double atr_tp_multiplier = 3.0;

// Indicators
int    atr_handle  = INVALID_HANDLE;
int    rsi_handle  = INVALID_HANDLE;
double atr_buffer[];

// Logging File Handle
int    log_handle  = INVALID_HANDLE;

// ------------------------------------------------------------------
// 3. LOGGER UTILITY
// ------------------------------------------------------------------
void LogMsg(string message)
  {
   // 1. Print to Experts Log
   Print(message);
   
   // 2. Write to physical .log file
   if(log_handle != INVALID_HANDLE)
     {
      string time_str = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
      FileWrite(log_handle, time_str + " | " + message);
      FileFlush(log_handle); // Ensure it's written immediately
     }
  }

void InitLogger()
  {
   string filename = "QuantTrader_" + _Symbol + "_" + TimeToString(TimeLocal(), TIME_DATE) + ".log";
   StringReplace(filename, ".", "");
   // Open file for appending text
   log_handle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_TXT|FILE_ANSI);
   
   if(log_handle != INVALID_HANDLE)
     {
      FileSeek(log_handle, 0, SEEK_END);
      LogMsg("=================================================");
      LogMsg("EA STARTING / INITIALIZED");
      LogMsg("=================================================");
     }
   else
     {
      Print("WARNING: Failed to create log file: ", filename);
     }
  }

void CloseLogger()
  {
   if(log_handle != INVALID_HANDLE)
     {
      LogMsg("EA SHUTDOWN.");
      FileClose(log_handle);
      log_handle = INVALID_HANDLE;
     }
  }

// ------------------------------------------------------------------
// 4. JSON LOADER FUNCTION
// ------------------------------------------------------------------
bool LoadConfiguration()
  {
   CJAVal config;
   string config_data = ReadFile(InpConfigPath);
   if(config_data == "") return false;
   
   if(config.Deserialize(config_data))
     {
      // Load thresholds and risk parameters automatically from Python Export
      buy_threshold     = config["buy_threshold"].ToDbl();
      sell_threshold    = config["sell_threshold"].ToDbl();
      atr_sl_multiplier = config["atr_sl"].ToDbl();
      atr_tp_multiplier = config["atr_tp"].ToDbl();
      
      PrintFormat("Loaded Config | Buy: %.2f | Sell: %.2f | SL: %.1fx | TP: %.1fx", 
                  buy_threshold, sell_threshold, atr_sl_multiplier, atr_tp_multiplier);
     }
   else 
     {
      Print("Failed to parse model_config.json");
      return false;
     }
     
   CJAVal feature_order;
   string feature_data = ReadFile(InpFeaturePath);
   if(feature_data == "") return false;
   
   if(feature_order.Deserialize(feature_data))
     {
      // Dynamically set the number of features!
      num_features = feature_order["features"].Size();
      PrintFormat("Loaded Feature Order | Detected %d Features.", num_features);
     }
   else
     {
      Print("Failed to parse feature_order.json");
      return false;
     }

   return true;
  }

string ReadFile(string file_path)
  {
   int handle = FileOpen(file_path, FILE_READ|FILE_TXT);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open file: ", file_path);
      return "";
     }
   string result = "";
   while(!FileIsEnding(handle))
     {
      result += FileReadString(handle) + "\n";
     }
   FileClose(handle);
   return result;
  }

// ------------------------------------------------------------------
// 5. INITIALIZATION (OnInit)
// ------------------------------------------------------------------
int OnInit()
  {
   InitLogger();
   LogMsg("Initializing Dynamic QuantTrader ONNX EA...");
   
   // 1. Load Dynamic Configuration from JSON
   if(!LoadConfiguration())
     {
      LogMsg("CRITICAL: Failed to load JSON configs. Halting EA.");
      return(INIT_FAILED);
     }
     
   if(num_features <= 0)
     {
      LogMsg("CRITICAL: num_features is 0. Check feature_order.json.");
      return(INIT_FAILED);
     }
   
   atr_handle = iATR(_Symbol, PERIOD_D1, 14);
   if(atr_handle == INVALID_HANDLE) 
     {
      LogMsg("Error: Failed to get ATR handle.");
      return(INIT_FAILED);
     }
   ArraySetAsSeries(atr_buffer, true);

   if(InpEnableONNX)
     {
      onnx_handle = OnnxCreate(InpModelPath, ONNX_DEFAULT);
      if(onnx_handle == INVALID_HANDLE)
        {
         LogMsg(StringFormat("CRITICAL: Failed to load ONNX model from %s. Error: %d", InpModelPath, GetLastError()));
         return(INIT_FAILED);
        }
      LogMsg(StringFormat("ONNX System Ready. Model requires %d features.", num_features));
     }

   return(INIT_SUCCEEDED);
  }

// ------------------------------------------------------------------
// 6. DEINITIALIZATION (OnDeinit)
// ------------------------------------------------------------------
void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE)
     {
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
     }
   IndicatorRelease(atr_handle);
   CloseLogger();
  }

// ------------------------------------------------------------------
// 7. MAIN EXECUTION LOOP (OnTick)
// ------------------------------------------------------------------
void OnTick()
  {
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_D1, 0);
   if(current_time == last_time) return; 
   
   // --- A. RISK LAYER: Spread Filter ---
   long current_spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(current_spread > InpMaxSpread) 
     {
      LogMsg(StringFormat("Rejected: Spread too high (%d > %d)", current_spread, InpMaxSpread));
      return;
     }

   // --- B. FEATURE LAYER ---
   float features[];
   ArrayResize(features, num_features);
   
   // (Feature Generation Logic Here to fill the array...)
   // ...
   
   if(!ValidateFeatures(features)) 
     {
      LogMsg("Invalid Features Detected (NaN/INF). NO TRADE.");
      return;
     }

   // --- C. ONNX INFERENCE LAYER ---
   float probability_buy = 0.5; 
   if(InpEnableONNX && onnx_handle != INVALID_HANDLE)
     {
      vectorf output_data(1);
      bool success = OnnxRun(onnx_handle, ONNX_NO_CONVERSION, features, output_data);
      if(success) probability_buy = output_data[0];
      else LogMsg(StringFormat("ONNX Inference Error: %d", GetLastError()));
     }
     
   // --- D. SIGNAL LAYER (Using Dynamic Thresholds) ---
   int signal = 0;
   if(probability_buy >= buy_threshold) signal = 1;
   else if(probability_buy <= sell_threshold) signal = -1;
   
   if(signal == 0) 
     {
      // Optional: LogMsg(StringFormat("Prob: %.2f (No Signal)", probability_buy));
      return;
     }

   // --- E. POSITION SIZING ENGINE ---
   double sl_points = CalculateATRStopLoss();
   double volume = CalculateLotSize(sl_points);
   
   last_time = current_time; 
   LogMsg(StringFormat("Signal Detected! Dir: %d | Prob: %.2f", signal, probability_buy));
   ExecuteTrade(signal, volume, sl_points);
  }

// ------------------------------------------------------------------
// 8. HELPER FUNCTIONS 
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
   // Uses the dynamically loaded multiplier!
   return atr * atr_sl_multiplier / _Point; 
  }

double CalculateLotSize(double sl_points)
  {
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (InpRiskPercent / 100.0);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   
   if(tick_value <= 0 || sl_points <= 0) return 0.01;
   
   double lot = risk_amount / (sl_points * tick_value);
   return NormalizeDouble(lot, 2);
  }

void ExecuteTrade(int signal, double lot, double sl_points)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = 0, tp = 0;
   
   if(signal == 1) // BUY
     {
      sl = ask - (sl_points * _Point);
      // Dynamic TP based on JSON
      tp = ask + ((sl_points / atr_sl_multiplier) * atr_tp_multiplier * _Point);
      PrintFormat("Executing BUY | Lot: %.2f | SL: %.5f | TP: %.5f", lot, sl, tp);
     }
   else if(signal == -1) // SELL
     {
      sl = bid + (sl_points * _Point);
      tp = bid - ((sl_points / atr_sl_multiplier) * atr_tp_multiplier * _Point);
      PrintFormat("Executing SELL | Lot: %.2f | SL: %.5f | TP: %.5f", lot, sl, tp);
     }
  }
