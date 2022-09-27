// Omits a warning of an unused value by assigning it to _.
public func unused(_ any: Any...) {
        for x in any {
                _ = x
        }
}
