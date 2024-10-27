function BerghD(F::StackyFan,divlist::Array{Int64,1}, Dlist::Array{Int64,1})
    X=deepcopy(F)
    slicedRayMatrix=getRays(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end
    divisors = Vector{Vector{QQFieldElem}}([])
    for i in [1:1:size(Dlist,1);]
        if Dlist[i] == 1
            push!(divisors,slicedRayMatrix[i])
        end
    end
    for divisor in divisors
        while(true)
            div= deepcopy(div)
            slicedRayMatrix=getRays(X.fan)
            # Find the cones with maximal divisorial index in X
            subdivTargetCones=minMaxIndexAlongDiv(X,div,divisor)
            # If there arde no such cones, the algorithm terminates
            if subdivTargetCones==nothing
                break
            end
            blowupList=Array{Array{QQFieldElem,1},1}[]
            # Iterate through cones with maximal divisorial index
            for cone in subdivTargetCones
                # Add each cone's ray representation to blowupList
                push!(blowupList,newRowMinors(slicedRayMatrix, cone))
            end
            for raycone in blowupList
                indices=Int64[]
                slicedRayMatrix=getRays(X.fan)
                for ray in raycone
                    push!(indices,findall(x->x==ray,slicedRayMatrix)[1])
                end
                # Sort indices in ascending order
                cone=sort(indices)
                if size(cone,1)==1
                    div[slicedRayMatrix[cone[1]]]=1
                else
                    exceptional=findStackyBarycenter(cone,X)
                    # perform the blowup
                    X=stackyBlowup(X,cone, exceptional)
                    # convert exceptional ray to its primitive form
                    div[Vector{QQFieldElem}(normalize(exceptional))]=1
                end
            end
        end
    end
    return X, div
end

function subStackyFan(F::StackyFan, divisor::Vector{QQFieldElem})
    divisorCones = Vector{Vector{Int64}}([])
    nondivisorCones = Vector{Vector{Int64}}([])
    maximalCones = getMaximalCones(F.fan)
    for cone in maximalCones
        if divisor in newRowMinors(getRays(X.fan), cone)
            push!(divisorCones, cone)
        else
            push!(nondivisorCones, cone)
        end
    end
    newFan = normal_toric_variety(IncidenceMatrix(divisorCones),listToMatrix(getRays(F.fan)))
    newStackyFan = StackyFan(newFan,F.stacks)
    return newStackyFan, nondivisorCones
end

function BerghDmodi(F::StackyFan,divlist::Array{Int64,1}, Dlist::Array{Int64,1})
    X=deepcopy(F)
    slicedRayMatrix=getRays(X.fan)
    div=Dict{Vector{QQFieldElem}, Int64}()
    # Populate dictionary of divisorial rays (div) from divlist
    for i in 1:size(slicedRayMatrix,1)
        div[slicedRayMatrix[i]]=divlist[i]
    end
    divisors = Vector{Vector{QQFieldElem}}([])
    for i in [1:1:size(Dlist,1);]
        if Dlist[i] == 1
            push!(divisors,slicedRayMatrix[i])
        end
    end
    for divisor in divisors
        Xsub, nondivisorCones = subStackyFan(X,divisor)
        Xsub, divlist = BerghC(X,divlist)
        divisorCones = getMaximalCones(Xsub.fan)
        finalCones = map(x -> Vector{Int64}(x), append!(divisorCones, nondivisorCones))
        X = makeStackyFan(listToMatrix(getRays(Xsub.fan)),finalCones,X.stacks)
    end
    return X,divlist
end
