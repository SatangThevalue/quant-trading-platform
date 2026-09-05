//+------------------------------------------------------------------+
//|                                  QuantTrader_Daily_Income_Pro.mq5 |
//|                        Copyright 2026, Satang AI Trading Project |
//|                                              https://github.com/ |
//+------------------------------------------------------------------+
#property copyright "Satang AI Quant Platform"
#property link      "https://github.com/SatangThevalue/quant-trading-platform"
#property version   "3.00"
#property description "Professional Daily Income Generator for EURUSD (1H)."
#property description "Fully automated 24/7 with Institutional Risk & Time Filters."

#include <Trade\Trade.mqh>
#include <JAson.mqh>
CTrade trade;

// ------------------------------------------------------------------
// 1. INPUTS
// ------------------------------------------------------------------
sinput string   InpModelPath      = "models/production/EURUSD_model_v2.0_1h.onnx";
sinput string   InpConfigPath     = "models/production/EURUSD_model_config_v2.0_1h.json";
sinput string   InpFeaturePath    = "models/production/EURUSD_feature_order_v2.0_1h.json";
input  bool     InpEnableONNX     = true;                                       
input  double   InpRiskPercent    = 1.0; 

sinput string   RiskSettings      = "--- Institutional Risk Management ---";
input  double   InpMaxDailyDrawdown = 2.0;   // Max Daily Loss % (Stop trading if equity drops this much today)
input  double   InpMaxSpreadPip   = 1.5;     // Max Spread allowed to open trade

sinput string   TimeSettings      = "--- Institutional Time Management ---";
input  bool     InpCloseOnFriday  = true;    // Close all open positions on Friday before market close
input  int      InpFridayCloseHour= 21;      // Hour to close trades on Friday (Broker Time)
input  bool     InpAvoidRollover  = true;    // Pause trading during 23:55 - 00:05 to avoid swap/spread spikes

long   onnx_handle = INVALID_HANDLE;
double start_of_day_equity = 0.0;
int    current_day = -1;

int    num_features = 38; // Will be overridden by JSON
double buy_threshold = 0.65;
double sell_threshold = 0.35;

int OnInit()
  {
   trade.SetExpertMagicNumber(198404);
   
   // Load AI Model
   if(InpEnableONNX)
     {
      onnx_handle = OnnxCreate(InpModelPath, ONNX_DEFAULT);
      if(onnx_handle == INVALID_HANDLE)
        {
         Print("CRITICAL: Failed to load ONNX model!");
         return(INIT_FAILED);
        }
     }
     
   Print("Daily Income Pro Initialized. 24/7 Institutional Safeguards Enabled.");
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   if(onnx_handle != INVALID_HANDLE) OnnxRelease(onnx_handle);
  }

//+------------------------------------------------------------------+
//| Check Institutional Safeguards (Time & Risk)                     |
//+------------------------------------------------------------------+
bool PassesInstitutionalSafeguards()
  {
   MqlDateTime tm;
   TimeCurrent(tm);
   
   // 1. New Day Equity Tracking for Daily Drawdown
   if(tm.day_of_year != current_day)
     {
      start_of_day_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      current_day = tm.day_of_year;
     }
     
   // 2. Daily Drawdown Limit Check
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_dd_pct = ((start_of_day_equity - current_equity) / start_of_day_equity) * 100.0;
   
   if(daily_dd_pct >= InpMaxDailyDrawdown)
     {
      Print("SAFEGUARD: Daily Drawdown Limit Hit (-", daily_dd_pct, "%). Trading Halted for today.");
      return false; // Block trading
     }
     
   // 3. Avoid Rollover Spread Spikes (23:55 to 00:05)
   if(InpAvoidRollover)
     {
      if((tm.hour == 23 && tm.min >= 55) || (tm.hour == 0 && tm.min <= 5))
        {
         // Print("SAFEGUARD: Avoiding Rollover Spread Spike.");
         return false;
        }
     }
     
   // 4. Spread Check Filter
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double max_spread = InpMaxSpreadPip * 10 * SymbolInfoDouble(_Symbol, SYMBOL_POINT); // Convert Pips to absolute
   if(spread > max_spread)
     {
      Print("SAFEGUARD: Spread too high: ", spread, " > ", max_spread);
      return false;
     }
     
   return true;
  }

//+------------------------------------------------------------------+
//| Friday Close Logic                                               |
//+------------------------------------------------------------------+
void CheckFridayClose()
  {
   if(!InpCloseOnFriday) return;
   
   MqlDateTime tm;
   TimeCurrent(tm);
   
   if(tm.day_of_week == 5 && tm.hour >= InpFridayCloseHour)
     {
      // If we have open positions, close them all to avoid weekend gaps
      if(PositionsTotal() > 0)
        {
         Print("SAFEGUARD: Friday Market Close Approaching. Closing all positions.");
         for(int i = PositionsTotal()-1; i >= 0; i--)
           {
            ulong ticket = PositionGetTicket(i);
            if(PositionGetString(POSITION_SYMBOL) == _Symbol)
              {
               trade.PositionClose(ticket);
              }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Main Execution Loop                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Execute Friday checks regardless of safeguards
   CheckFridayClose();
   
   // Ensure we only process signals once per H1 bar
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_H1, 0);
   if(current_time == last_time) return;
   
   // Hard Block if safeguards fail
   if(!PassesInstitutionalSafeguards()) return;
   
   // --- ONNX INFERENCE LOGIC (Simplified for Demo) ---
   // float probability_up = RunONNXModel();
   float probability_up = 0.5; // Placeholder
   
   if(probability_up >= buy_threshold)
     {
      // Calculate ATR Position Sizing
      // trade.Buy(...);
     }
   else if(probability_up <= sell_threshold)
     {
      // trade.Sell(...);
     }
     
   last_time = current_time;
  }