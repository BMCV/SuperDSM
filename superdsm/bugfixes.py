BUGFIX_ENABLED  = 1
BUGFIX_DISABLED = 0
BUGFIX_DISABLED_CRITICAL = -1


def is_enabled(bugfix):
    if bugfix < 0:
        raise AssertionError(
            'Critical bugfix is disabled, aborting. ' + \
            'Either enable the bugfix (set to BUGFIX_ENABLED) or mark it as non-critical (set to BUGFIX_DISABLED).')
    elif bugfix == 0:
        return False  ## disable the bugfix and proceed
    else:
        return True   ## enable the bugfix


#BUGFIX_20210408A = BUGFIX_DISABLED
