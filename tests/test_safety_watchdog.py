import time
from raspberry_pi.hardware.motor_driver import MockMotorDriver
from raspberry_pi.safety.watchdog import MotorWatchdog

def test_watchdog_start_motor():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    # Motor starts successfully
    assert watchdog.start_motor('mouth', 1.0, start_time=10.0) is True
    assert driver.mouth == 1.0
    
    # Auto stop enforcement
    watchdog.check_and_enforce(current_time=11.1)
    assert driver.mouth == 0.0
    
    # Cooldown enforcement
    assert watchdog.start_motor('mouth', 1.0, start_time=11.2) is False
    
    # Start after cooldown
    assert watchdog.start_motor('mouth', 1.0, start_time=11.7) is True
    assert driver.mouth == 1.0

def test_watchdog_emergency_stop():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    watchdog.start_motor('body', 1.0, start_time=10.0)
    assert driver.body == 1.0
    
    watchdog.emergency_stop()
    assert driver.body == 0.0
    assert 'body' in watchdog.cooldown_motors

def test_watchdog_restart_active_motor():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    watchdog.start_motor('mouth', 1.0, start_time=10.0)
    # Restart at 10.5
    watchdog.start_motor('mouth', 1.0, start_time=10.5)
    
    # Check enforcement at 11.1 (should stop because original start time 10.0 is used)
    watchdog.check_and_enforce(current_time=11.1)
    assert driver.mouth == 0.0
    assert 'mouth' in watchdog.cooldown_motors

def test_watchdog_graceful_stop():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    watchdog.start_motor('head', 1.0, start_time=10.0)
    # Graceful stop
    watchdog.start_motor('head', 0.0, start_time=10.5)
    
    assert driver.head == 0.0
    assert 'head' not in watchdog.active_motors
    assert 'head' not in watchdog.cooldown_motors

def test_watchdog_stop_motor_preserves_cooldown():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    watchdog.cooldown_motors['head'] = 10.0
    watchdog.stop_motor('head')
    
    assert 'head' in watchdog.cooldown_motors

def test_watchdog_stop_all_preserves_cooldown():
    driver = MockMotorDriver()
    watchdog = MotorWatchdog(driver, max_runtime=1.0, cooldown=0.5)
    
    watchdog.cooldown_motors['mouth'] = 15.0
    watchdog.stop_all()
    
    assert 'mouth' in watchdog.cooldown_motors
    assert watchdog.cooldown_motors['mouth'] == 15.0
