"""
    BerghA(F::StackyFan,D::Array{Int64,1})

Given a stacky fan F and a vector of booleans D representing the distinguished structure, returns a smooth stacky fan where the distinguished rays are independent.

The algorithm is adapted from Daniel Bergh's [paper on destackification](https://arxiv.org/abs/1409.5713). In brief, it identifies non-smooth cones containing at least one distinguished ray, finds interior points in those cones, and subdivides at those points through a series of stacky barycentric subdivisions.

# Examples
```jldoctest AlgA

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,1]))
[ 5 ,  2 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,0]));
[ 5 ,  5 ,  1 ,  2 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 2 5],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[0,1]))
[ 1 ,  5 ,  5 ,  2 ,  5 ,  1 ,  5 ,  10 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[4 1; 7 9],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,0]))
[ 609 ,  29 ,  1 ,  174 ,  29 ,  1740 ,  1218 ,  58 ,  145 ,  1044 ,  1044 ,  290 ,  290 ,  145 ,  406 ,  348 ,  261 ,  14616 ,  14616 ,  609 ,  3480 ,  870 ,  609 ,  174 ,  609 ,  9744 ,  14616 ,  1218 ,  58 ,  145 ,  435 ,  725 ,  1305 ,  1740 ,  6960 ,  3480 ,  870 ,  3480 ,  58464 ,  6090 ,  3480 ,  1392 ,  696 ,  1044 ,  2088 ,  261 ,  174 ,  261 ,  609 ,  406 ,  609 ,  609 ,  406 ,  609 ,  1218 ,  812 ,  1218 ,  2088 ,  261 ,  1044 ,  1160 ,  1740 ,  1740 ,  870 ,  1305 ,  1305 ,  14616 ,  10440 ,  145 ,  435 ,  609 ,  116 ,  580 ,  290 ,  580 ,  1740 ,  3480 ,  3480 ,  261 ,  522 ,  261 ,  522 ,  522 ,  1218 ,  1218 ,  1218 ,  1218 ,  2436 ,  2436 ,  4176 ,  2088 ,  3480 ,  2610 ,  29232 ,  6090 ,  1218 ,  290 ,  145 ,  290 ,  12180 ,  261 ,  522 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[323 179; 44 135],INPUT_CONES=[[0,1]]);

julia> F=addStackStructure(X,[1,1]);

julia> stackyWeights(BerghA(F,[1,1]))
[ 491602456800 ,  49160245680 ,  294961474080 ,  468192816 ,  12173013216 ,  73038079296 ,  12173013216 ,  97384105728 ,  73038079296 ,  196640982720 ,  131093988480 ,  156064272 ,  936385632 ,  468192816 ,  196640982720 ,  49160245680 ,  9832049136 ,  292152317184 ,  5899229481600 ,  3932819654400 ,  589922948160 ,  393281965440 ,  12173013216 ,  8115342144 ,  12173013216 ,  5899229481600 ,  589922948160 ,  737403685200 ,  51126655507200 ,  292152317184 ,  51126655507200 ,  5899229481600 ,  5899229481600 ,  589922948160 ,  589922948160 ,  196640982720 ,  2949614740800 ,  5899229481600 ,  294961474080 ,  589922948160 ,  196640982720 ,  589922948160 ,  393281965440 ,  936385632 ,  24346026432 ,  24346026432 ,  11798458963200 ,  1179845896320 ,  737403685200 ,  1474807370400 ,  2949614740800 ,  5899229481600 ,  589922948160 ,  11798458963200 ,  1179845896320 ,  2949614740800 ,  1474807370400 ,  5899229481600 ,  102253311014400 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 3; 4 5 6; 2 3 1],INPUT_CONES=[[0,1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 28 ,  21 ,  84 ,  28 ,  84 ,  84 ,  42 ,  84 ,  168 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0 2; 2 1 1; 5 3 9],INPUT_CONES=[[0,1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 4 ,  4 ,  8 ,  4 ,  4 ,  8 ]

julia> X=Polymake.fulton.NormalToricVariety(INPUT_RAYS=[1 0; 4 3; 1 5],INPUT_CONES=[[0,1],[1,2]]);

julia> F=addStackStructure(X,[1,1,1]);

julia> stackyWeights(BerghA(F,[1,1,1]))
[ 4 ,  34 ,  6 ,  68 ,  34 ,  68 ,  34 ,  6 ,  12 ]
```
"""
function BerghA(F::StackyFan,D::Array{Int64,1};verbose::Bool=false)
    if verbose==true
        println("==algorithm is running in verbose mode==")
        println(" ")
        println("=======")
    end
    X=deepcopy(F)
    
    #rayMatrix=convert(Array{Int64,2},Array(Polymake.common.primitive(X.fan.RAYS)))
    rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
    rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
    
    coneList=getCones(X.fan)
    dim=size(rayMatrix,2)
    numRays=size(rayMatrix,1)
    
    #check if the vector D has length equal to the number of rays in F
    if numRays != size(D,1)
        error("length of vector representing distinguished structure does not agree with number of rays in stacky fan.")
    end
    
    #A0: initialization
    i=0
    while(true)
        rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
        rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
        
        numRays=size(rayMatrix,1)
        coneList=getMaximalCones(X.fan)
        
        #A1: Find S the set of cones that contain a distinguised ray and an interior lattice point 
        #Note: cones in S are 1-indexed.
        S=filter(cone->distinguishedAndIntPoint(cone,rayMatrix,D),coneList)
        # If S is empty, the program terminates.
        print(S)
        if S==[]
            break
        end
        
        #A2 - find extremal cones
        Smax=extremalCones(S,rayMatrix,D)
        Smax = map(x->sort(x),Smax)
        
        #Print information on the number of extremal cones, their number of non-distinguished rays, and their multiplicity
        #The algorithm is structured to first reduce the number of non-distinguished rays in extremal cones, and then reduce the multiplicity of said cones,
            #so this information can be used to track the algorithm's progress
        if verbose==true
            Smaxcount=size(Smax,1)
            println("Number of extremal cones: $Smaxcount")
            testCone=Smax[1]
            c1=convertToIncidence(testCone,numRays)
            nonDist=size(testCone,1)-dot(c1,D)
            mult=coneMultiplicity(coneConvert(testCone,rayMatrix))
            println("Maximal non-distinguished rays and multiplicity: $nonDist, $mult")
        end
        
        #A2 - find interior points in Smax
        intPoints=[]
        for cone in Smax
            #C=rowMinors(rayMatrix,cone)
            C=coneConvert(cone,rayMatrix)
            coneIntPoints=interiorPoints(C)
            for point in coneIntPoints
               push!(intPoints,(point,cone)) #the point is stored as a tuple along with the cone containing it
            end
        end

        #A2 - find stacky points (in terms of coefficients) derived from interior points
        P=Array{Int64,1}[]
        for (point,cone) in intPoints
            stackyWeight = Array{Int64}(Polymake.common.primitive(stackyWeights(X)))
            stackyPoint=coneRayDecomposition(cone,rayMatrix,point,stackyWeight) #each interior point is rewritten as a string of coefficients
                #corresponding to its representation as a sum of stacky rays
            push!(P,stackyPoint)
        end
        

        
        #A2 - find element of P such that the sum of the coefficients corresponding to distinguished rays is minimal.
            #This invariant does not produce a unique ray, so there is a degree of arbitrary selection.
        psi=minimalByDist(P,D)
        #if verbose==true
            #println("Psi: $psi")
        #end
        
        #A3 - perform root construction
        X=rootConstructionDistinguishedIndices(X,D,psi)

        

        #A3 - modify psi with respect to root construction
        for i in 1:length(psi)
            if D[i]==1 && psi[i]>1
                psi[i]=1
            end
        end
        
        # A5 - perform repeated stacky barycentric star subdivision with respect to psi.
        while(count(x->x>0,psi)>1)
            
            supportCone=findall(x->x!=0,psi)
            #find the stacky barycenter of that cone, which becomes the exceptional (blowup) ray
            exceptional=findStackyBarycenter(supportCone,X)
            code_rays =  map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            # Track the indices of distinguished rays
            D_pairs = map((x,y) -> (x,y), code_rays, D)
            D_Dict = Dict(D_pairs)
            # Track psi as a linear combination of the generators
            psiPairs = map((x,y) -> (x,y), code_rays,psi)
            psiDict = Dict(psiPairs)
            
            X= stackyBlowup(X,supportCone,exceptional)


            G=gcd(exceptional) #since the blowup ray may not be primitive, it is made primitive and then assigned a stacky value so its stacky form is unchanged.
            primExcep=Array{QQFieldElem}(Polymake.common.primitive(exceptional))
                        
            D_Dict[primExcep]=1
            psiDict[primExcep]=1
            
            newRays=map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            newD=Int64[]
            newpsi=Int64[]
            
            for ray in newRays
                E=ray
                excepCode=primExcep
                push!(newD,D_Dict[E])
                #A4 - modify ps
                if E==excepCode
                    push!(newpsi,1)
                elseif psiDict[E]>1
                    push!(newpsi,psiDict[E]-1)
                else
                    push!(newpsi,0)
                end
            end
            psi=newpsi
            D=newD
        end
        if verbose==true
            println("=======")
        end
        i+=1
    end
    if verbose==true
        println("Number of steps: $i")
    end
    return X,D
end

function BerghAmod(F::StackyFan,D::Array{Int64,1};verbose::Bool=false)
    if verbose==true
        println("==algorithm is running in verbose mode==")
        println(" ")
        println("=======")
    end
    X=deepcopy(F)
    
    #rayMatrix=convert(Array{Int64,2},Array(Polymake.common.primitive(X.fan.RAYS)))
    rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
    rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
    
    coneList=getCones(X.fan)
    dim=size(rayMatrix,2)
    numRays=size(rayMatrix,1)
    
    #check if the vector D has length equal to the number of rays in F
    if numRays != size(D,1)
        error("length of vector representing distinguished structure does not agree with number of rays in stacky fan.")
    end
    
    #A0: initialization
    i=0
    stackyModifications = []
    while(true)
        rayMatrixQQ = listToMatrix(map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan)))
        rayMatrix = map(x->Int(numerator(x)),rayMatrixQQ)
        
        numRays=size(rayMatrix,1)
        coneList=getCones(X.fan)
        
        #A1: Find S the set of cones that contain a distinguised ray and an interior lattice point 
        #Note: cones in S are 1-indexed.
        S=filter(cone->distinguishedAndIntPoint(cone,rayMatrix,D),coneList)
        # If S is empty, the program terminates.
        if S==[]
            break
        end
        
        #A2 - find extremal cones
        Smax=extremalCones(S,rayMatrix,D)
        Smax = map(x->sort(x),Smax)
        
        #Print information on the number of extremal cones, their number of non-distinguished rays, and their multiplicity
        #The algorithm is structured to first reduce the number of non-distinguished rays in extremal cones, and then reduce the multiplicity of said cones,
            #so this information can be used to track the algorithm's progress
        if verbose==true
            Smaxcount=size(Smax,1)
            println("Number of extremal cones: $Smaxcount")
            testCone=Smax[1]
            c1=convertToIncidence(testCone,numRays)
            nonDist=size(testCone,1)-dot(c1,D)
            mult=coneMultiplicity(coneConvert(testCone,rayMatrix))
            println("Maximal non-distinguished rays and multiplicity: $nonDist, $mult")
        end
        
        #A2 - find interior points in Smax
        intPoints=[]
        for cone in Smax
            #C=rowMinors(rayMatrix,cone)
            C=coneConvert(cone,rayMatrix)
            coneIntPoints=interiorPoints(C)
            for point in coneIntPoints
               push!(intPoints,(point,cone)) #the point is stored as a tuple along with the cone containing it
            end
        end
        
        #A2 - find stacky points (in terms of coefficients) derived from interior points
        P=Array{Int64,1}[]
        for (point,cone) in intPoints
            stackyWeight = Array{Int64}(Polymake.common.primitive(stackyWeights(X)))
            stackyPoint=coneRayDecomposition(cone,rayMatrix,point,stackyWeight) #each interior point is rewritten as a string of coefficients
                #corresponding to its representation as a sum of stacky rays
            push!(P,stackyPoint)
        end
        

        
        #A2 - find element of P such that the sum of the coefficients corresponding to distinguished rays is minimal.
            #This invariant does not produce a unique ray, so there is a degree of arbitrary selection.
        psi=minimalByDist(P,D)
        #if verbose==true
            #println("Psi: $psi")
        #end
        
        #A3 - perform root construction
        X=rootConstructionDistinguishedIndices(X,D,psi)

        ###############################################################################################

        all_rays = getRays(X.fan)
        dist_rays = []
        for i in [1:1:size(all_rays, 1);]
            if D[i]==1
                push!(dist_rays,(all_rays[i],psi[i]))
            end
        end
        pair = ("root",dist_rays)
        push!(stackyModifications, pair)

        ##############################################################################################

        #A3 - modify psi with respect to root construction
        for i in 1:length(psi)
            if D[i]==1 && psi[i]>1
                psi[i]=1
            end
        end
        
        # A5 - perform repeated stacky barycentric star subdivision with respect to psi.
        while(count(x->x>0,psi)>1)
            
            supportCone=findall(x->x!=0,psi)
            #find the stacky barycenter of that cone, which becomes the exceptional (blowup) ray
            exceptional=findStackyBarycenter(supportCone,X)
            code_rays =  map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            # Track the indices of distinguished rays
            D_pairs = map((x,y) -> (x,y), code_rays, D)
            D_Dict = Dict(D_pairs)
            # Track psi as a linear combination of the generators
            psiPairs = map((x,y) -> (x,y), code_rays,psi)
            psiDict = Dict(psiPairs)

            ###########################################################################

            all_rays = getRays(X.fan)
            raysOfCone = [all_rays[i] for i in supportCone]

            pair = ("blowup",raysOfCone)
            push!(stackyModifications, pair)
            
            #########################################################################################
            
            X= stackyBlowup(X,supportCone,exceptional)

            G=gcd(exceptional) #since the blowup ray may not be primitive, it is made primitive and then assigned a stacky value so its stacky form is unchanged.
            primExcep=Array{QQFieldElem}(Polymake.common.primitive(exceptional))
                        
            D_Dict[primExcep]=1
            psiDict[primExcep]=1
            
            newRays=map(x->Array{QQFieldElem}(Polymake.common.primitive(x)),getRays(X.fan))
            newD=Int64[]
            newpsi=Int64[]
            
            for ray in newRays
                E=ray
                excepCode=primExcep
                push!(newD,D_Dict[E])
                #A4 - modify ps
                if E==excepCode
                    push!(newpsi,1)
                elseif psiDict[E]>1
                    push!(newpsi,psiDict[E]-1)
                else
                    push!(newpsi,0)
                end
            end
            psi=newpsi
            D=newD
        end
        if verbose==true
            println("=======")
        end
        i+=1
    end
    if verbose==true
        println("Number of steps: $i")
    end
    return X,D,stackyModifications
end

function getIndex(L, v)
    i= 1
    while L[i] != v
        i+=1
    end
    return i
end

function manualBerghA(F::StackyFan,D::Array{Int64,1}, manual)
    X = deepcopy(F)
    for instruction in manual
        if instruction[1] == "root"
            rootConstruction = instruction[2]
            all_rays = getRays(X.fan)
            dist_ray_indices = []
            for pair in rootConstruction
                distIndex = getIndex(all_rays,pair[1])
                push!(dist_ray_indices, (distIndex,pair[2]))
            end
            full_indices = [0 for i in [1:1:size(all_rays,1);]]
            for pair in dist_ray_indices
                ind = pair[1]
                c = pair[2]
                full_indices[ind] = c
            end
            X = rootConstructionDistinguishedIndices(X,D,full_indices)
        else
            blowup = instruction[2]
            #find the support cone of the blowup
            all_rays = getRays(X.fan)
            supportCone = Array{Int64}([])
            for Ray in blowup
                ind = getIndex(all_rays,Ray)
                push!(supportCone, ind)
            end

            DistPairs = Dict(map((x,y) -> (x,y), all_rays, D))
            barycenter = findStackyBarycenter(supportCone,X)
            primBarycenter = Array{QQFieldElem}(Polymake.common.primitive(barycenter))
            X = stackyBlowup(X,supportCone,barycenter)
            DistPairs[primBarycenter]= 1
            new_all_rays = getRays(X.fan)
            
            newD = [0 for i in [1:1:size(new_all_rays,1);]]
            for i in [1:1:size(new_all_rays,1);]
                theRay = new_all_rays[i]
                newD[i] = DistPairs[theRay]
            end
            D = newD
        end
    end
    return X,D
end

