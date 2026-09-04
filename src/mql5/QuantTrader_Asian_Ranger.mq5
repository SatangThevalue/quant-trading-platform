//+------------------------------------------------------------------+
//|                                     QuantTrader_Asian_Ranger.mq5 |
//|                        Copyright 2026, Satang AI Trading Project |
//|                                              https://github.com/ |
//+------------------------------------------------------------------+
#property copyright "Satang AI Quant Platform"
#property link      "https://github.com/SatangThevalue/quant-trading-platform"
#property version   "1.00"
#property description "Inspired by Top Asian Session Scalpers on MQL5 Market."
#property description "Focuses on Mean-Reversion during low volatility periods."

#include <Trade\Trade.mqh>
CTrade trade;

// ------------------------------------------------------------------
// 1. INPUTS
// ------------------------------------------------------------------
sinput string   InpModelPath      = "models/production/GBPUSD_model_v1.0.onnx";
input  bool     InpEnableONNX     = true;                                       
input  double   InpBuyThreshold   = 0.65; // High threshold for counter-trend
input  double   InpSellThreshold  = 0.35; 

// The "Asian Ranger" Secret Sauce (Time Filters)
sinput string   TimeSettings      = "--- Time Filter (Broker Time) ---";
input  int      InpStartHour      = 23;   // Start looking for trades (e.g., 23:00 GMT+2)
input  int      InpEndHour        = 3;    // Stop looking for trades (e.g., 03:00 GMT+2)
input  bool     InpCloseOnFriday  = true; // Prevent weekend gaps

// Grid/Recovery (Use with Extreme Caution - Kept small to emulate market EAs)
sinput string   GridSettings      = "--- Smart Recovery ---";
input  bool     InpUseRecovery    = false; // Default off, true mimics popular EAs
input  double   InpGridStepATR    = 2.0;   // Distance to next grid in ATR
input  int      InpMaxGridTrades  = 3;     // Max trades in a basket
input  double   InpGridMultiplier = 1.5;   // Martingale multiplier

// Volatility Filter
sinput string   VolSettings       = "--- Volatility Filter ---";
input  double   InpMaxATR         = 0.0030;// Do not trade if ATR > 30 pips (Market is too crazy for Asian Scalping)

long   onnx_handle = INVALID_HANDLE;
int    atr_handle  = INVALID_HANDLE;
double atr_buffer[];

int OnInit()
  {
   trade.SetExpertMagicNumber(198402); // Asian Ranger Magic Number
   
   atr_handle = iATR(_Symbol, PERIOD_H1, 14);
   if(atr_handle == INVALID_HANDLE) return(INIT_FAILED);
   ArraySetAsSeries(atr_buffer, true);
   
   // ... [ONNX Initialization skipped for brevity, identical to V2] ...
   Print("Asian Ranger Initialized.");
   return(INIT_SUCCEEDED);
  }

void OnTick()
  {
   // 1. Time Filter (The core of Night Scalping)
   MqlDateTime tm;
   TimeCurrent(tm);
   
   bool is_trading_session = false;
   if(InpStartHour > InpEndHour)
     {
      // Crosses midnight
      if(tm.hour >= InpStartHour || tm.hour < InpEndHour) is_trading_session = true;
     }
   else
     {
      // Same day
      if(tm.hour >= InpStartHour && tm.hour < InpEndHour) is_trading_session = true;
     }
     
   if(InpCloseOnFriday && tm.day_of_week == 5 && tm.hour >= 20)
     {
      // Close all trades before weekend
      // CloseAllPositions();
      return;
     }

   if(!is_trading_session) return;

   // 2. Volatility Filter (Check if market is quiet enough)
   CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
   if(atr_buffer[0] > InpMaxATR) return; // Too volatile to mean-revert
   
   // 3. Grid Management (Check if we need to average down)
   // if(InpUseRecovery) ManageGrid();
   
   // 4. ONNX Inference
   // float probability_buy = 0.5;
   // ... run ONNX ...
   
   // 5. Execution (Mean Reversion)
   // If probability is high, but we only trade counter to Bollinger Bands/VWAP
   // (Logic simplified for brevity)
  }
