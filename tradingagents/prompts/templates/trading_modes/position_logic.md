Current Position: {current_position}

Swing Trading Position Transition Logic:
- If Current Position: LONG
  - Signal: LONG -> Hold swing position, monitor daily with multi-day horizon
  - Signal: NEUTRAL -> Close LONG position at next favorable exit point
  - Signal: SHORT -> Close LONG position and enter SHORT swing position

- If Current Position: SHORT
  - Signal: SHORT -> Hold swing position, monitor daily with multi-day horizon
  - Signal: NEUTRAL -> Close SHORT position at next favorable exit point
  - Signal: LONG -> Close SHORT position and enter LONG swing position

- If Current Position: NEUTRAL (no open position)
  - Signal: LONG -> Enter LONG swing position based on multi-timeframe setup
  - Signal: SHORT -> Enter SHORT swing position based on multi-timeframe setup
  - Signal: NEUTRAL -> Stay in cash, wait for clear swing setup with confluence
