WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'


def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False
