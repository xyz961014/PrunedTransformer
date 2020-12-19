def param_in(p, params):
    for param in params:
        if p.equal(param):
            return True
    else:
        return False
 
