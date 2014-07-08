function value = parseOption(option, valuename, defval )
    exists = isfield(option, valuename);
    if exists
        value = option.(valuename);
        return;
    end
    
    if nargin > 2
        value = defval;
    else
        value = 0;
    end
        
end

